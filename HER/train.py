import sys
from pathlib import Path
from typing import Optional

from tqdm import trange

import torch
import torch.optim as optim

import gymnasium
import gymnasium_robotics

import numpy as np

from RLAlg.alg.ddpg import DDPG
from RLAlg.buffer.her import HERReplayBuffer, env_reward_fn
from RLAlg.nn.steps import DeterministicContinuousPolicyStep
from RLAlg.utils import set_seed_everywhere
from RLAlg.logger import WandbLogger

try:
    from .model import Actor, Critic
except ImportError:
    from model import Actor, Critic

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_utils import default_checkpoint_path, episode_metrics, reset_finished_envs, save_checkpoint

gymnasium.register_envs(gymnasium_robotics)

class Trainer:
    def __init__(self, env_name:str, env_num:int, seed:int=0):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        
        set_seed_everywhere(self.seed)

        self.env_name = env_name
        self.env_num = env_num
        # Disable gymnasium 1.x NEXT_STEP autoreset: finished envs are reset explicitly after
        # recording the real final observation, so no terminal->reset transition is stored.
        self.envs = gymnasium.vector.SyncVectorEnv(
            [lambda: self.setup_env(env_name) for _ in range(env_num)],
            autoreset_mode=gymnasium.vector.AutoresetMode.DISABLED,
        )

        self.max_steps = self.envs.envs[0].spec.max_episode_steps
        self.rollout_steps = self.max_steps
        self.max_buffer_steps = 100000

        self.max_action = torch.from_numpy(self.envs.single_action_space.high).float().to(self.device)
        obs_space = self.envs.single_observation_space["observation"].shape
        goal_space = self.envs.single_observation_space["desired_goal"].shape
        action_space = self.envs.single_action_space.shape
        
        obs_dim = np.prod(obs_space)
        goal_dim = np.prod(goal_space)
        if isinstance(self.envs.single_action_space, gymnasium.spaces.Discrete):
            action_dim = self.envs.single_action_space.n
        elif isinstance(self.envs.single_action_space, gymnasium.spaces.Box):
            action_dim = np.prod(self.envs.single_action_space.shape)

        self.actor = Actor(obs_dim+goal_dim, action_dim, [128, 128], self.max_action).to(self.device)
        self.critic = Critic(obs_dim+goal_dim, action_dim, [128, 128]).to(self.device)
        self.actor_target = Actor(obs_dim+goal_dim, action_dim, [128, 128], self.max_action).to(self.device)
        self.critic_target = Critic(obs_dim+goal_dim, action_dim, [128, 128]).to(self.device)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        for param in self.actor_target.parameters():
            param.requires_grad = False
            
        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # Relabels 1 - 1/(1 + replay_k) of sampled goals with goals achieved later in the same episode.
        self.replay_buffer = HERReplayBuffer(
            env_num, self.max_buffer_steps, device=self.device,
            reward_fn=env_reward_fn(self.envs), strategy="future", replay_k=4,
        )
        self.replay_buffer.create_storage_space("observations", obs_space, torch.float32)
        self.replay_buffer.create_storage_space("next_observations", obs_space, torch.float32)
        self.replay_buffer.create_storage_space("next_achieved_goal", goal_space, torch.float32)
        self.replay_buffer.create_storage_space("desired_goal", goal_space, torch.float32)
        self.replay_buffer.create_storage_space("actions", action_space, torch.float32)
        self.replay_buffer.create_storage_space("rewards", (), torch.float32)
        # "episode_end" (terminated | truncated) delimits episodes for relabeling;
        # "terminated" alone masks the Bellman bootstrap so timeouts still bootstrap.
        self.replay_buffer.create_storage_space("episode_end", (), torch.bool)
        self.replay_buffer.create_storage_space("terminated", (), torch.float32)

        self.batch_keys = ["observations", "next_observations", "actions",
                           "rewards", "terminated", "desired_goal"]
        
        self.gamma = 0.99
        self.regularization_weight = 0.0
        self.tau = 0.005
        self.max_grad_norm = 1.0
        
        self.global_step = 0
        WandbLogger.init_project("RLDemos", f"HER-{env_name}-{seed}")
        
        
    def setup_env(self, env_name:str, mode:Optional[str]=None) -> gymnasium.wrappers.RecordEpisodeStatistics:
        env = gymnasium.make(env_name, render_mode=mode)
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)

        return env
    
    @torch.no_grad()
    def get_action(self, obs:np.ndarray, goal:np.ndarray, random:bool=False):
        obs = torch.from_numpy(obs).float().to(self.device)
        goal = torch.from_numpy(goal).float().to(self.device)
        
        obs_dict = {
            "observation": obs,
            "desired_goal": goal
        }
        
        actor_step:DeterministicContinuousPolicyStep  = self.actor(obs_dict)
        
        if random:
            action = actor_step.mean.uniform_(-1, 1) * self.max_action
        else:
            action = actor_step.mean
            action += torch.randn_like(action) * 0.1
            action = torch.clamp(action, -self.max_action, self.max_action)
            
        return action
    
    def rollout(self, random:bool=False):
        obs_dict = self.obs_dict
        for i in range(self.rollout_steps):
            self.global_step += self.env_num
            obs = obs_dict["observation"]
            goal = obs_dict["desired_goal"]
            
            action = self.get_action(obs, goal, random)
            next_obs_dict, reward, terminated, truncated, info = self.envs.step(action.cpu().numpy())
            
            next_obs = next_obs_dict["observation"]
            next_achieved_goal = next_obs_dict["achieved_goal"]
            
            record = {
                "observations": obs,
                "next_observations": next_obs,
                "next_achieved_goal": next_achieved_goal,
                "desired_goal": goal,
                "actions": action,
                "rewards": reward,
                "episode_end": np.logical_or(terminated, truncated),
                "terminated": terminated
            }
            
            self.replay_buffer.add_records(record)

            obs_dict = reset_finished_envs(self.envs, next_obs_dict, terminated, truncated)
            
            episode_info = episode_metrics(info)
            if episode_info:
                WandbLogger.log_metrics(episode_info, self.global_step)
        
        self.obs_dict = obs_dict
        
    def update(self, num_iteration:int, batch_size:int):
        policy_loss_buffer = []
        critic_loss_buffer = []
        q_buffer = []
        q_target_buffer = []
        
        for _ in range(num_iteration):
            batch = self.replay_buffer.sample_batch(self.batch_keys, batch_size)

            obs_batch = {
                "observation": batch["observations"].to(self.device),
                "desired_goal": batch["desired_goal"].to(self.device),
            }
            next_obs_batch = {
                "observation": batch["next_observations"].to(self.device),
                "desired_goal": batch["desired_goal"].to(self.device),
            }
            action_batch = batch["actions"].to(self.device)
            # CriticHead squeezes Q to [B]; keep rewards/masks [B] to avoid [B, B] broadcasting.
            reward_batch = batch["rewards"].to(self.device)
            done_batch = batch["terminated"].to(self.device)
            
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss_dict = DDPG.compute_critic_loss(
                self.actor_target, self.critic, self.critic_target,
                observations=obs_batch, actions=action_batch, rewards=reward_batch,
                next_observations=next_obs_batch, terminated=done_batch, gamma=self.gamma,
            )
            critic_loss = critic_loss_dict["loss"]
            q = critic_loss_dict["q"]
            q_target = critic_loss_dict["q_target"]
            critic_loss.backward()
            self.critic_optimizer.step()

            for param in self.critic.parameters():
                param.requires_grad = False

            self.actor_optimizer.zero_grad(set_to_none=True)
            policy_loss_dict = DDPG.compute_policy_loss(
                self.actor, self.critic, observations=obs_batch,
                regularization_weight=self.regularization_weight,
            )
            policy_loss = policy_loss_dict["loss"]
            policy_loss.backward()
            self.actor_optimizer.step()

            for param in self.critic.parameters():
                param.requires_grad = True

            DDPG.update_target_param(self.actor, self.actor_target, self.tau)
            DDPG.update_target_param(self.critic, self.critic_target, self.tau)
            
            policy_loss_buffer.append(policy_loss.item())
            critic_loss_buffer.append(critic_loss.item())
            q_buffer.append(q.item())
            q_target_buffer.append(q_target.item())
            
        avg_policy_loss = np.mean(policy_loss_buffer)
        avg_critic_loss = np.mean(critic_loss_buffer)
        avg_q = np.mean(q_buffer)
        avg_q_target = np.mean(q_target_buffer)
        
        train_info = {
            "update/avg_policy_loss": avg_policy_loss,
            "update/avg_critic_loss": avg_critic_loss,
            "update/avg_q": avg_q,
            "update/avg_q_target": avg_q_target,
        }

        WandbLogger.log_metrics(train_info, self.global_step)
                
    def train(self, total_frames:int, num_iteration:int, batch_size:int):
        if total_frames <= 0:
            raise ValueError(f"total_frames must be positive, got {total_frames}.")
        self.obs_dict, _ = self.envs.reset(seed=[i+self.seed for i in range(self.envs.num_envs)])
        start_step = self.global_step
        target_step = start_step + total_frames
        frames_per_rollout = self.env_num * self.rollout_steps
        num_rollouts = max(1, (total_frames + frames_per_rollout - 1) // frames_per_rollout)
        warmup_frames = total_frames // 10

        for _ in trange(num_rollouts):
            remaining_frames = target_step - self.global_step
            if remaining_frames <= 0:
                break

            collected_frames = self.global_step - start_step
            random = collected_frames <= warmup_frames
            rollout_steps = min(
                self.rollout_steps,
                max(1, (remaining_frames + self.env_num - 1) // self.env_num),
            )
            original_rollout_steps = self.rollout_steps
            self.rollout_steps = rollout_steps
            try:
                self.rollout(random)
            finally:
                self.rollout_steps = original_rollout_steps
            self.update(num_iteration, batch_size)
        
        self.envs.close()
        
if __name__ == "__main__":
    trainer = Trainer("FetchReachDense-v4", 20, seed=0)
    
    trainer.train(total_frames=100_000, num_iteration=20, batch_size=500)
    save_checkpoint(trainer, default_checkpoint_path("HER", "FetchReachDense-v4", 0), algorithm="HER", env_name="FetchReachDense-v4", seed=0)
