import sys
from pathlib import Path
from typing import Union, Optional

from tqdm import trange

import torch
import torch.optim as optim

import gymnasium
import numpy as np

from RLAlg.alg.sac import SAC
from RLAlg.buffer.replay_buffer import ReplayBuffer
from RLAlg.nn.steps import StochasticContinuousPolicyStep, DiscretePolicyStep, ValueStep
from RLAlg.utils import set_seed_everywhere
from RLAlg.logger import WandbLogger, MetricsTracker

from model import Actor, Critic, Encoder

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

        self.max_action = torch.from_numpy(self.envs.single_action_space.high).float().to(self.device)
        obs_space = self.envs.single_observation_space.shape
        action_space = self.envs.single_action_space.shape
        
        obs_dim = np.prod(obs_space)
        if isinstance(self.envs.single_action_space, gymnasium.spaces.Discrete):
            action_dim = self.envs.single_action_space.n
        elif isinstance(self.envs.single_action_space, gymnasium.spaces.Box):
            action_dim = np.prod(self.envs.single_action_space.shape)

        self.actor_encoder = Encoder(obs_dim, [128, 128]).to(self.device)
        self.actor = Actor(self.actor_encoder.feature_dim, action_dim, [128, 128], self.max_action).to(self.device)
        self.critic_encoder = Encoder(obs_dim, [128, 128]).to(self.device)
        self.critic = Critic(self.critic_encoder.feature_dim, action_dim, [128, 128]).to(self.device)
        self.critic_target_encoder = Encoder(obs_dim, [128, 128]).to(self.device)
        self.critic_target = Critic(self.critic_target_encoder.feature_dim, action_dim, [128, 128]).to(self.device)
        
        self.critic_target_encoder.load_state_dict(self.critic_encoder.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        for param in self.critic_target_encoder.parameters():
            param.requires_grad = False
        
        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.actor_optimizer = optim.Adam(list(self.actor_encoder.parameters()) + list(self.actor.parameters()), lr=3e-4)
        self.critic_optimizer = optim.Adam(list(self.critic_encoder.parameters()) + list(self.critic.parameters()), lr=3e-4)
        
        self.replay_buffer = ReplayBuffer(env_num, self.max_buffer_steps, device=self.device)
        self.replay_buffer.create_storage_space("observations", obs_space, torch.float32)
        self.replay_buffer.create_storage_space("next_observations", obs_space, torch.float32)
        self.replay_buffer.create_storage_space("actions", action_space, torch.float32)
        self.replay_buffer.create_storage_space("rewards", (), torch.float32)
        self.replay_buffer.create_storage_space("dones", (), torch.float32)

        self.batch_keys = ["observations", "next_observations", "actions", "rewards", "dones"]
        
        self.gamma = 0.99
        self.alpha = 0.2
        self.regularization_weight = 0.0
        self.tau = 0.005
        self.max_grad_norm = 1.0
        
        self.global_step = 0
        WandbLogger.init_project("RLDemos", f"SAC-{env_name}-{seed}")
        
        
    def setup_env(self, env_name:str, mode:Optional[str]=None) -> gymnasium.wrappers.RecordEpisodeStatistics:
        env = gymnasium.make(env_name, render_mode=mode)
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)

        return env
    
    @torch.no_grad()
    def get_action(self, obs:np.ndarray, random:bool=False):
        obs = torch.from_numpy(obs).float().to(self.device)
        feat = self.actor_encoder(obs)
        actor_step:Union[StochasticContinuousPolicyStep, DiscretePolicyStep]  = self.actor(feat)
        
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
            next_obs, reward, terminate, timeout, info = self.envs.step(action.cpu().numpy())
            done = terminate  # True termination only; timeouts still bootstrap.
            record = {
                "observations": obs,
                "next_observations": next_obs,
                "actions": action,
                "rewards": reward,
                "dones": done
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
        q1_buffer = []
        q2_buffer = []
        q_target_buffer = []
        
        for _ in range(num_iteration):
            batch = self.replay_buffer.sample_batch(self.batch_keys, batch_size)
            obs_batch = batch["observations"].to(self.device)
            next_obs_batch = batch["next_observations"].to(self.device)
            action_batch = batch["actions"].to(self.device)
            reward_batch = batch["rewards"].to(self.device)
            done_batch = batch["dones"].to(self.device)
            
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_feat_batch = self.critic_encoder(obs_batch)
            
            with torch.no_grad():
                next_actor_feat_batch = self.actor_encoder(next_obs_batch)
                next_critic_feat_batch = self.critic_target_encoder(next_obs_batch)
            # The critic loss never evaluates the actor on the current state, so only critic features
            # are needed there; the actor sees actor features of the next state.
            critic_loss_dict = SAC.compute_critic_loss(
                self.actor,
                self.critic,
                self.critic_target,
                observations=critic_feat_batch,
                critic_observations=critic_feat_batch,
                actions=action_batch,
                rewards=reward_batch,
                next_observations=next_actor_feat_batch,
                next_critic_observations=next_critic_feat_batch,
                terminated=done_batch,
                alpha=self.alpha,
                gamma=self.gamma,
            )
            
            critic_loss = critic_loss_dict["loss"]
            q1 = critic_loss_dict["q1"]
            q2 = critic_loss_dict["q2"]
            q_target = critic_loss_dict["q_target"]
            
            critic_loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            for param in self.critic_encoder.parameters():
                param.requires_grad = False
                
            for param in self.critic.parameters():
                param.requires_grad = False

            self.actor_optimizer.zero_grad(set_to_none=True)
            
            actor_feat_batch = self.actor_encoder(obs_batch)
            critic_feat_batch = self.critic_encoder(obs_batch)
            
            policy_loss_dict = SAC.compute_policy_loss(
                self.actor, self.critic, observations=actor_feat_batch, critic_observations=critic_feat_batch,
                alpha=self.alpha, regularization_weight=self.regularization_weight,
            )
            
            policy_loss = policy_loss_dict["loss"]
            
            policy_loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            for param in self.critic_encoder.parameters():
                param.requires_grad = True
            for param in self.critic.parameters():
                param.requires_grad = True

            SAC.update_target_param(self.critic_encoder, self.critic_target_encoder, self.tau)
            SAC.update_target_param(self.critic, self.critic_target, self.tau)
            
            policy_loss_buffer.append(policy_loss.item())
            critic_loss_buffer.append(critic_loss.item())
            q1_buffer.append(q1.item())
            q2_buffer.append(q2.item())
            q_target_buffer.append(q_target.item())

        avg_policy_loss = np.mean(policy_loss_buffer)
        avg_critic_loss = np.mean(critic_loss_buffer)
        avg_q1 = np.mean(q1_buffer)
        avg_q2 = np.mean(q2_buffer)
        avg_q_target = np.mean(q_target_buffer)
        
        train_info = {
            "update/avg_policy_loss": avg_policy_loss,
            "update/avg_critic_loss": avg_critic_loss,
            "update/avg_q1": avg_q1,
            "update/avg_q2": avg_q2,
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
    
    trainer.train(total_frames=100_000, num_iteration=12, batch_size=500)
    save_checkpoint(trainer, default_checkpoint_path("SAC_asym", "HalfCheetah-v5", 100), algorithm="SAC_asym", env_name="HalfCheetah-v5", seed=100)
    
