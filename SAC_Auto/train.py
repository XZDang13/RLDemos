import sys
from pathlib import Path
from typing import Optional, Union

from tqdm import trange

import torch
import torch.optim as optim

import gymnasium
import numpy as np

from RLAlg.alg.sac import SAC
from RLAlg.buffer.replay_buffer import ReplayBuffer
from RLAlg.nn.steps import StochasticContinuousPolicyStep, DiscretePolicyStep
from RLAlg.nn.layers import NormPosition
from RLAlg.utils import set_seed_everywhere
from RLAlg.logger import WandbLogger

from model import Actor, Critic

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_utils import default_checkpoint_path, save_checkpoint, episode_metrics, reset_finished_envs

class Trainer:
    def __init__(self, env_name:str, env_num:int, seed:int=0):
        self.seed = seed
        set_seed_everywhere(self.seed)


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env_name = env_name
        self.env_num = env_num
        # Disable autoreset so the real terminal observation is stored; finished envs are reset explicitly.
        self.envs = gymnasium.vector.SyncVectorEnv([lambda: self.setup_env(env_name) for _ in range(env_num)],
                                                   autoreset_mode=gymnasium.vector.AutoresetMode.DISABLED)

        self.max_steps = self.envs.envs[0].spec.max_episode_steps
        self.rollout_steps = 50
        self.max_buffer_steps = 100000

        if not isinstance(self.envs.single_action_space, gymnasium.spaces.Box):
            raise ValueError("SAC_Auto demo only supports continuous Box action spaces.")

        self.max_action = torch.from_numpy(self.envs.single_action_space.high).float().to(self.device)
        obs_space = self.envs.single_observation_space.shape
        action_space = self.envs.single_action_space.shape
        
        obs_dim = np.prod(obs_space)
        action_dim = np.prod(self.envs.single_action_space.shape)

        self.actor = Actor(obs_dim, action_dim, [128, 128], self.max_action, norm_position=NormPosition.PRE).to(self.device)
        self.critic = Critic(obs_dim, action_dim, [128, 128], norm_position=NormPosition.PRE).to(self.device)
        self.critic_target = Critic(obs_dim, action_dim, [128, 128], norm_position=NormPosition.PRE).to(self.device)
        
        self.critic_target.load_state_dict(self.critic.state_dict())

        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.replay_buffer = ReplayBuffer(env_num, self.max_buffer_steps, device=self.device)
        self.replay_buffer.create_storage_space("observations", obs_space, torch.float32)
        self.replay_buffer.create_storage_space("next_observations", obs_space, torch.float32)
        self.replay_buffer.create_storage_space("actions", action_space, torch.float32)
        self.replay_buffer.create_storage_space("rewards", (), torch.float32)
        self.replay_buffer.create_storage_space("terminated", (), torch.float32)
        self.replay_buffer.create_storage_space("truncated", (), torch.float32)
        self.replay_buffer.create_storage_space("episode_end", (), torch.float32)

        self.batch_keys = ["observations", "next_observations", "actions", "rewards", "terminated"]
        
        self.gamma = 0.99
        self.log_alpha = torch.tensor(np.log(0.2), dtype=torch.float32, device=self.device, requires_grad=True)
        self.target_entropy = -float(action_dim)
        self.regularization_weight = 0.0
        self.tau = 0.005
        self.max_grad_norm = 1.0
        self.alpha = self.log_alpha.exp().detach()
        
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        
        self.global_step = 0
        WandbLogger.init_project("RLDemos", f"SAC_Auto-{env_name}-{seed}")
        
        
    def setup_env(self, env_name:str, mode:Optional[str]=None) -> gymnasium.wrappers.RecordEpisodeStatistics:
        env = gymnasium.make(env_name, render_mode=mode)
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)

        return env
    
    @torch.no_grad()
    def get_action(self, obs:np.ndarray, random:bool=False):
        obs = torch.from_numpy(obs).float().to(self.device)
        actor_step:Union[StochasticContinuousPolicyStep, DiscretePolicyStep]  = self.actor(obs)
        
        if random:
            action = actor_step.action.uniform_(-1, 1) * self.max_action
        else:
            action = actor_step.action
            
        return action
    
    def rollout(self, random:bool=False):
        obs = self.obs
        for i in range(self.rollout_steps):
            self.global_step += self.env_num
            
            action = self.get_action(obs, random)
            action_cpu = action.cpu()
            next_obs, reward, terminate, timeout, info = self.envs.step(action_cpu.numpy())
            terminated = terminate.astype(np.float32)
            truncated = timeout.astype(np.float32)
            episode_end = np.logical_or(terminate, timeout).astype(np.float32)
            record = {
                "observations": obs,
                "next_observations": next_obs,
                "actions": action_cpu,
                "rewards": reward,
                "terminated": terminated,
                "truncated": truncated,
                "episode_end": episode_end,
            }
            
            self.replay_buffer.add_records(record)
            
            obs = reset_finished_envs(self.envs, next_obs, terminate, timeout)
            
            episode_info = episode_metrics(info)
            if episode_info:
                WandbLogger.log_metrics(episode_info, self.global_step)
        
        self.obs = obs
        
    def update(self, num_iteration:int, batch_size:int):
        policy_loss_buffer = []
        critic_loss_buffer = []
        alpha_loss_buffer = []
        alpha_buffer = []
        q1_buffer = []
        q2_buffer = []
        q_target_buffer = []
        
        for _ in range(num_iteration):
            batch = self.replay_buffer.sample_batch(self.batch_keys, batch_size)
            obs_batch = batch["observations"].to(self.device)
            next_obs_batch = batch["next_observations"].to(self.device)
            action_batch = batch["actions"].to(self.device)
            reward_batch = batch["rewards"].to(self.device)
            terminated_batch = batch["terminated"].to(self.device)
            alpha = self.log_alpha.exp().detach()
            
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss_dict = SAC.compute_critic_loss(
                self.actor,
                self.critic,
                self.critic_target,
                observations=obs_batch,
                actions=action_batch,
                rewards=reward_batch,
                next_observations=next_obs_batch,
                terminated=terminated_batch,  # True termination only; timeouts still bootstrap.
                alpha=alpha,
                gamma=self.gamma,
            )
            
            critic_loss = critic_loss_dict["loss"]
            q1 = critic_loss_dict["q1"]
            q2 = critic_loss_dict["q2"]
            q_target = critic_loss_dict["q_target"]
                
            if not torch.isfinite(critic_loss):
                continue

            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            for param in self.critic.parameters():
                param.requires_grad = False

            self.actor_optimizer.zero_grad(set_to_none=True)
            policy_loss_dict = SAC.compute_policy_loss(
                self.actor,
                self.critic,
                observations=obs_batch.detach(),
                alpha=alpha,
                regularization_weight=self.regularization_weight
            )
            
            policy_loss = policy_loss_dict["loss"]
                
            policy_loss.backward()
            self.actor_optimizer.step()

            for param in self.critic.parameters():
                param.requires_grad = True
                
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss_dict = SAC.compute_alpha_loss(
                self.actor,
                log_alpha=self.log_alpha,
                observations=obs_batch.detach(),
                target_entropy=self.target_entropy
            )
            alpha_loss = alpha_loss_dict["loss"]
                
            if not torch.isfinite(alpha_loss):
                continue
            alpha_loss.backward()
            torch.nn.utils.clip_grad_norm_([self.log_alpha], self.max_grad_norm)
            self.alpha_optimizer.step()
            
            with torch.no_grad():
                self.log_alpha.clamp_(min=-20.0, max=2.0)
                self.alpha = self.log_alpha.exp().detach()

            SAC.update_target_param(self.critic, self.critic_target, self.tau)
            
            policy_loss_buffer.append(policy_loss.item())
            critic_loss_buffer.append(critic_loss.item())
            alpha_loss_buffer.append(alpha_loss.item())
            alpha_buffer.append(self.alpha.item())
            
            q1_buffer.append(q1.item())
            q2_buffer.append(q2.item())
            q_target_buffer.append(q_target.item())
            
        if not policy_loss_buffer:
            return

        avg_policy_loss = np.mean(policy_loss_buffer)
        avg_critic_loss = np.mean(critic_loss_buffer)
        avg_alpha_loss = np.mean(alpha_loss_buffer)
        avg_alpha = np.mean(alpha_buffer)
        
        train_info = {
            "update/avg_policy_loss": avg_policy_loss,
            "update/avg_critic_loss": avg_critic_loss,
            "update/avg_alpha_loss": avg_alpha_loss,
            "update/avg_alpha": avg_alpha,
        }
        
        train_info["update/avg_q1"] = np.mean(q1_buffer)
        train_info["update/avg_q2"] = np.mean(q2_buffer)
        train_info["update/avg_q_target"] = np.mean(q_target_buffer)

        WandbLogger.log_metrics(train_info, self.global_step)
                
    def train(self, total_frames:int, num_iteration:int, batch_size:int):
        if total_frames <= 0:
            raise ValueError(f"total_frames must be positive, got {total_frames}.")
        self.obs, _ = self.envs.reset(seed=[i+self.seed for i in range(self.envs.num_envs)])
        start_step = self.global_step
        target_step = start_step + total_frames
        frames_per_rollout = self.env_num * self.rollout_steps
        num_rollouts = max(1, (total_frames + frames_per_rollout - 1) // frames_per_rollout)
        warmup_frames = total_frames // 5

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
    trainer = Trainer("HalfCheetah-v5", 20, seed=100)
    
    trainer.train(total_frames=1_000_000, num_iteration=25, batch_size=500)
    save_checkpoint(trainer, default_checkpoint_path("SAC_Auto", "HalfCheetah-v5", 100), algorithm="SAC_Auto", env_name="HalfCheetah-v5", seed=100)
    
