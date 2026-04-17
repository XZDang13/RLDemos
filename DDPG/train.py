from typing import Union, Optional

from tqdm import trange

import torch
import torch.optim as optim

import gymnasium
import numpy as np

try:
    import gymnasium_robotics
except ModuleNotFoundError:
    gymnasium_robotics = None

from RLAlg.alg.ddpg import DDPG
from RLAlg.buffer.replay_buffer import ReplayBuffer
from RLAlg.nn.steps import DeterministicContinuousPolicyStep, ValueStep
from RLAlg.utils import set_seed_everywhere
from RLAlg.logger import WandbLogger, MetricsTracker

from model import Actor, Critic

if gymnasium_robotics is not None:
    gymnasium.register_envs(gymnasium_robotics)

class Trainer:
    def __init__(self, env_name:str, env_num:int, seed:int=0, n_step:int=1):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        
        set_seed_everywhere(self.seed)

        self.env_name = env_name
        self.env_num = env_num
        self.envs = gymnasium.vector.SyncVectorEnv([lambda: self.setup_env(env_name) for _ in range(env_num)])

        self.max_steps = self.envs.envs[0].spec.max_episode_steps
        self.rollout_steps = 50
        self.max_buffer_steps = 100000

        self.max_action = torch.from_numpy(self.envs.single_action_space.high).float().to(self.device)
        obs_space = self.envs.single_observation_space.shape
        action_space = self.envs.single_action_space.shape
        
        obs_dim = np.prod(obs_space)
        if isinstance(self.envs.single_action_space, gymnasium.spaces.Discrete):
            action_dim = self.envs.single_action_space.n
        elif isinstance(self.envs.single_action_space, gymnasium.spaces.Box):
            action_dim = np.prod(self.envs.single_action_space.shape)

        self.actor = Actor(obs_dim, action_dim, [128, 128], self.max_action).to(self.device)
        self.critic = Critic(obs_dim, action_dim, [128, 128]).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim, [128, 128], self.max_action).to(self.device)
        self.critic_target = Critic(obs_dim, action_dim, [128, 128]).to(self.device)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        for param in self.actor_target.parameters():
            param.requires_grad = False
            
        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.replay_buffer = ReplayBuffer(env_num, self.max_buffer_steps, device=self.device)
        self.replay_buffer.create_storage_space("observations", obs_space, torch.float32)
        self.replay_buffer.create_storage_space("next_observations", obs_space, torch.float32)
        self.replay_buffer.create_storage_space("actions", action_space, torch.float32)
        self.replay_buffer.create_storage_space("rewards", (), torch.float32)
        self.replay_buffer.create_storage_space("dones", (), torch.float32)

        self.batch_keys = ["observations", "next_observations", "actions", "rewards", "dones"]
        
        self.gamma = 0.99
        if n_step <= 0:
            raise ValueError(f"n_step must be positive, got {n_step}.")
        self.n_step = n_step
        self.sample_batch_kwargs = {
            "n_step": self.n_step,
            "gamma": self.gamma,
            "reward_key": "rewards",
            "terminal_key": "dones",
            "bootstrap_keys": ["next_observations"],
        }
        self.alpha = 0.2
        self.regularization_weight = 0.0
        self.tau = 0.005
        self.max_grad_norm = 1.0
        
        self.global_step = 0
        WandbLogger.init_project("RLDemos", f"DDPG-{env_name}-{seed}")
        
        
    def setup_env(self, env_name:str, mode:Optional[str]=None) -> gymnasium.wrappers.RecordEpisodeStatistics:
        env = gymnasium.make(env_name, render_mode=mode)
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)

        return env
    
    @torch.no_grad()
    def get_action(self, obs:np.ndarray, random:bool=False):
        obs = torch.from_numpy(obs).float().to(self.device)
        actor_step:DeterministicContinuousPolicyStep  = self.actor(obs)
        
        if random:
            action = actor_step.mean.uniform_(-1, 1) * self.max_action
        else:
            action = actor_step.mean
            action += torch.randn_like(action) * 0.1
            action = torch.clamp(action, -self.max_action, self.max_action)
            
        return action
    
    def rollout(self, random:bool=False):
        obs = self.obs
        for i in range(self.rollout_steps):
            self.global_step += self.env_num
            
            action = self.get_action(obs, random)
            next_obs, reward, terminate, timeout, info = self.envs.step(action.numpy())
            done = terminate | timeout
            
            record = {
                "observations": obs,
                "next_observations": next_obs,
                "actions": action,
                "rewards": reward,
                "dones": done
            }
            
            self.replay_buffer.add_records(record)
            
            obs = next_obs
            
            if "episode" in info:
                finished = info['episode']['_r']
                episode_info = {}
                episode_info['episode/mean_rewards'] = np.mean(info['episode']['r'][finished])
                episode_info['episode/mean_length'] = np.mean(info['episode']['l'][finished])
                
                WandbLogger.log_metrics(episode_info, self.global_step)
        
        self.obs = obs
        
    def update(self, num_iteration:int, batch_size:int):
        policy_loss_buffer = []
        critic_loss_buffer = []
        q_buffer = []
        q_target_buffer = []
        
        for _ in range(num_iteration):
            batch = self.replay_buffer.sample_n_step_batch(self.batch_keys, batch_size, **self.sample_batch_kwargs)
            obs_batch = batch["observations"].to(self.device)
            next_obs_batch = batch["next_observations"].to(self.device)
            action_batch = batch["actions"].to(self.device)
            reward_batch = batch["rewards"].to(self.device)
            done_batch = batch["dones"].to(self.device)
            discount_batch = batch.get("bootstrap_discount")
            if discount_batch is None:
                discount_batch = batch.get("n_step_discount")
            if discount_batch is not None:
                discount_batch = discount_batch.to(self.device)
            
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss_dict = DDPG.compute_critic_loss(
                self.actor_target,
                self.critic,
                self.critic_target,
                obs_batch,
                action_batch,
                reward_batch,
                next_obs_batch,
                done_batch,
                gamma=self.gamma,
                discount=discount_batch,
            )
            critic_loss = critic_loss_dict["loss"]
            q = critic_loss_dict["q"]
            q_target = critic_loss_dict["q_target"]
            critic_loss.backward()
            self.critic_optimizer.step()

            for param in self.critic.parameters():
                param.requires_grad = False

            self.actor_optimizer.zero_grad(set_to_none=True)
            policy_loss_dict = DDPG.compute_policy_loss(self.actor, self.critic, obs_batch.detach(), self.regularization_weight)
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
        self.obs, _ = self.envs.reset(seed=[i+self.seed for i in range(self.envs.num_envs)])
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
        
        
if __name__ == "__main__":
    trainer = Trainer("HalfCheetah-v5", 20, seed=0)
    
    trainer.train(total_frames=100_000, num_iteration=10, batch_size=500)
