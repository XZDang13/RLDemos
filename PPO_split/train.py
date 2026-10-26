import sys
from pathlib import Path
from typing import Union, Optional

from tqdm import trange

import torch
import torch.optim as optim

import gymnasium
import numpy as np

from RLAlg.alg.ppo import PPO
from RLAlg.buffer import ReplayBuffer, compute_gae
from RLAlg.nn.steps import StochasticContinuousPolicyStep, DiscretePolicyStep, ValueStep
from RLAlg.utils import set_seed_everywhere
from RLAlg.logger import WandbLogger, MetricsTracker

from model import ContinuousActor, DiscreteActor, Critic, Encoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_utils import default_checkpoint_path, reset_finished_envs, save_checkpoint

class Trainer:
    def __init__(self, env_name:str, env_num:int, seed:int=0):
        self.seed = seed
        set_seed_everywhere(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env_name = env_name
        self.env_num = env_num
        # Disable gymnasium 1.x NEXT_STEP autoreset: finished envs are reset explicitly after
        # recording the real final observation, so no terminal->reset transition is stored.
        self.envs = gymnasium.vector.SyncVectorEnv(
            [lambda: self.setup_env(env_name) for _ in range(env_num)],
            autoreset_mode=gymnasium.vector.AutoresetMode.DISABLED,
        )

        self.max_steps = self.envs.envs[0].spec.max_episode_steps
        self.rollout_steps = 50

        obs_space = self.envs.single_observation_space.shape
        action_space = self.envs.single_action_space.shape
        
        obs_dim = np.prod(obs_space)
        
        self.actor_encoder = Encoder(obs_dim, [128, 128]).to(self.device)
        self.critic_encoder = Encoder(obs_dim, [128, 128]).to(self.device)
        if isinstance(self.envs.single_action_space, gymnasium.spaces.Discrete):
            action_dim = self.envs.single_action_space.n
            self.actor = DiscreteActor(self.actor_encoder.feature_dim, action_dim, [128, 128]).to(self.device)
        elif isinstance(self.envs.single_action_space, gymnasium.spaces.Box):
            self.max_action = torch.from_numpy(self.envs.single_action_space.high).float().to(self.device)
            action_dim = np.prod(self.envs.single_action_space.shape)
            self.actor = ContinuousActor(self.actor_encoder.feature_dim, action_dim, [128, 128], self.max_action).to(self.device)

        self.critic = Critic(self.critic_encoder.feature_dim, [128, 128]).to(self.device)
        
        self.optimizer = optim.Adam(
            list(self.actor_encoder.parameters()) + list(self.critic_encoder.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters()), lr=3e-4
        )
        
        self.replay_buffer = ReplayBuffer(env_num, self.rollout_steps, device=self.device)
        self.replay_buffer.create_storage_space("observations", obs_space, torch.float32)
        self.replay_buffer.create_storage_space("actions", action_space, torch.float32)
        self.replay_buffer.create_storage_space("log_probs", (), torch.float32)
        self.replay_buffer.create_storage_space("rewards", (), torch.float32)
        self.replay_buffer.create_storage_space("values", (), torch.float32)
        self.replay_buffer.create_storage_space("terminated", (), torch.float32)
        self.replay_buffer.create_storage_space("truncated", (), torch.bool)
        
        self.batch_keys = ["observations", "actions", "log_probs", "rewards", "values", "returns", "advantages"]
        
        self.gamma = 0.99
        self.lambda_ = 0.95
        self.clip_ratio = 0.2
        self.regularization_weight = 0.0
        self.max_grad_norm = 1.0
        self.value_loss_weight = 0.5
        self.entropy_weight = 0.01
        
        self.global_step = 0   
        WandbLogger.init_project("RLDemos", f"PPO-{env_name}-{seed}")
        
    def setup_env(self, env_name:str, mode:Optional[str]=None) -> gymnasium.wrappers.RecordEpisodeStatistics:
        env = gymnasium.make(env_name, render_mode=mode)
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)

        return env
    
    @torch.no_grad()
    def get_action(self, obs:np.ndarray):
        obs = torch.from_numpy(obs).float().to(self.device)
        actor_feat = self.actor_encoder(obs)
        critic_feat = self.critic_encoder(obs)
        actor_step:Union[StochasticContinuousPolicyStep, DiscretePolicyStep]  = self.actor(actor_feat)
        value_step:ValueStep = self.critic(critic_feat)
        
        action = actor_step.action
        log_prob = actor_step.log_prob
        value = value_step.value
        
        return action, log_prob, value

    @torch.no_grad()
    def timeout_rewards(self, rewards:np.ndarray, next_obs:np.ndarray, terminated:np.ndarray, truncated:np.ndarray) -> np.ndarray:
        # Timeouts end the episode for GAE but bootstrap from V(real final observation).
        rewards = np.array(rewards, copy=True)
        timeouts = truncated & ~terminated
        if timeouts.any():
            final_obs = torch.from_numpy(next_obs[timeouts]).float().to(self.device)
            final_value = self.critic(self.critic_encoder(final_obs)).value
            rewards[timeouts] += self.gamma * final_value.cpu().numpy()
        return rewards
    
    def rollout(self):
        self.replay_buffer.reset()
        obs = self.obs
        for i in range(self.rollout_steps):
            self.global_step += self.env_num
            action, log_prob, value = self.get_action(obs)
            next_obs, reward, terminate, timeout, info = self.envs.step(action.cpu().numpy())
            record = {
                "observations": obs,
                "actions": action,
                "log_probs": log_prob,
                "rewards": self.timeout_rewards(reward, next_obs, terminate, timeout),
                "values": value,
                "terminated": terminate,
                "truncated": timeout
            }
            
            self.replay_buffer.add_records(record)
            
            obs = reset_finished_envs(self.envs, next_obs, terminate, timeout)
            
            if "episode" in info and "_episode" in info:
                finished = info["_episode"]
                if np.any(finished):
                    episode_info = {}
                    episode_info["episode/mean_rewards"] = np.mean(info["episode"]["r"][finished])
                    episode_info["episode/mean_length"] = np.mean(info["episode"]["l"][finished])
                    
                    WandbLogger.log_metrics(episode_info, self.global_step)
                
        self.obs = obs
        _, _, value = self.get_action(obs)
        buffer_steps = self.replay_buffer.current_size
        returns, advantages = compute_gae(
            rewards=self.replay_buffer.data["rewards"][:buffer_steps],
            values=self.replay_buffer.data["values"][:buffer_steps],
            terminated=self.replay_buffer.data["terminated"][:buffer_steps],
            last_values=value,
            gamma=self.gamma,
            lambda_=self.lambda_,
            truncated=self.replay_buffer.data["truncated"][:buffer_steps],
            # Timeout rewards already include gamma * V(final_obs).
            bootstrap_timeouts=False,
            )

        returns_full = torch.zeros_like(self.replay_buffer.data["rewards"])
        advantages_full = torch.zeros_like(self.replay_buffer.data["rewards"])
        returns_full[:buffer_steps] = returns
        advantages_full[:buffer_steps] = advantages

        self.replay_buffer.add_storage("returns", returns_full)
        self.replay_buffer.add_storage("advantages", advantages_full)
        
    def update(self, num_iteration:int, batch_size:int):
        policy_loss_buffer = []
        value_loss_buffer = []
        entropy_buffer = []
        kl_divergence_buffer = []
        
        for _ in range(num_iteration):
            for batch in self.replay_buffer.sample_batchs(self.batch_keys, batch_size):
                obs_batch = batch["observations"].to(self.device)
                action_batch = batch["actions"].to(self.device)
                log_prob_batch = batch["log_probs"].to(self.device)
                value_batch = batch["values"].to(self.device)
                return_batch = batch["returns"].to(self.device)
                advantage_batch = batch["advantages"].to(self.device)

                actor_feat_batch = self.actor_encoder(obs_batch)
                critic_feat_batch = self.critic_encoder(obs_batch)
                
                policy_loss_dict = PPO.compute_policy_loss(self.actor, log_probs_hat=log_prob_batch, observations=actor_feat_batch,
                                                           actions=action_batch, advantages=advantage_batch, clip_ratio=self.clip_ratio,
                                                           regularization_weight=self.regularization_weight)

                policy_loss = policy_loss_dict["loss"]
                entropy = policy_loss_dict["entropy"]
                kl_divergence = policy_loss_dict["kl_divergence"]

                value_loss_dict = PPO.compute_clipped_value_loss(self.critic, observations=critic_feat_batch, values_hat=value_batch,
                                                                returns=return_batch, clip_ratio=self.clip_ratio)
                
                value_loss = value_loss_dict["loss"]
                
                loss = policy_loss + value_loss * self.value_loss_weight - entropy * self.entropy_weight

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_encoder.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic_encoder.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                policy_loss_buffer.append(policy_loss.item())
                value_loss_buffer.append(value_loss.item())
                entropy_buffer.append(entropy.item())
                kl_divergence_buffer.append(kl_divergence.item())
                
        avg_policy_loss = np.mean(policy_loss_buffer)
        avg_value_loss = np.mean(value_loss_buffer)
        avg_entropy = np.mean(entropy_buffer)
        avg_kl_divergence = np.mean(kl_divergence_buffer)
        
        train_info = {
            "update/avg_policy_loss": avg_policy_loss,
            "update/avg_value_loss": avg_value_loss,
            "update/avg_entropy": avg_entropy,
            "update/avg_kl_divergence": avg_kl_divergence
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

        for _ in trange(num_rollouts):
            remaining_frames = target_step - self.global_step
            if remaining_frames <= 0:
                break

            rollout_steps = min(
                self.rollout_steps,
                max(1, (remaining_frames + self.env_num - 1) // self.env_num),
            )
            original_rollout_steps = self.rollout_steps
            self.rollout_steps = rollout_steps
            try:
                self.rollout()
            finally:
                self.rollout_steps = original_rollout_steps
            self.update(num_iteration, batch_size)
        
        
if __name__ == "__main__":
    trainer = Trainer("HalfCheetah-v5", 20, seed=100)
    
    trainer.train(total_frames=100_000, num_iteration=10, batch_size=500)
    save_checkpoint(trainer, default_checkpoint_path("PPO_split", "HalfCheetah-v5", 100), algorithm="PPO_split", env_name="HalfCheetah-v5", seed=100)
