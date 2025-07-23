from typing import Union, Optional

from tqdm import trange

import torch
import torch.optim as optim

import gymnasium
import numpy as np

from RLAlg.alg.ppo import PPO
from RLAlg.buffer.replay_buffer import ReplayBuffer, compute_gae
from RLAlg.nn.steps import StochasticContinuousPolicyStep, DiscretePolicyStep, ValueStep
from RLAlg.utils import set_seed_everywhere
from RLAlg.logger import WandbLogger

from model import ContinuousActor, DiscreteActor, Critic

class Trainer:
    def __init__(self, env_name:str, env_num:int, seed:int=0):
        self.seed = seed
        set_seed_everywhere(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env_name = env_name
        self.env_num = env_num
        self.envs = gymnasium.vector.SyncVectorEnv([lambda: self.setup_env(env_name) for _ in range(env_num)])

        self.max_steps = self.envs.envs[0].spec.max_episode_steps
        self.rollout_steps = self.max_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env_name = env_name
        self.envs = gymnasium.vector.SyncVectorEnv([lambda: self.setup_env(env_name) for _ in range(env_num)])

        self.max_steps = self.envs.envs[0].spec.max_episode_steps
        self.rollout_steps = self.max_steps

        self.max_action = torch.from_numpy(self.envs.single_action_space.high).float().to(self.device)
        obs_space = self.envs.single_observation_space.shape
        action_space = self.envs.single_action_space.shape
        
        obs_dim = np.prod(obs_space)
        if isinstance(self.envs.single_action_space, gymnasium.spaces.Discrete):
            action_dim = self.envs.single_action_space.n
            self.actor = DiscreteActor(obs_dim, action_dim, [128, 128]).to(self.device)
        elif isinstance(self.envs.single_action_space, gymnasium.spaces.Box):
            action_dim = np.prod(self.envs.single_action_space.shape)
            self.actor = ContinuousActor(obs_dim, action_dim, [128, 128], self.max_action).to(self.device)

        self.critic = Critic(obs_dim, [128, 128]).to(self.device)
        
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=3e-4
        )
        
        self.replay_buffer = ReplayBuffer(env_num, self.max_steps, device=self.device)
        self.replay_buffer.create_storage_space("observations", obs_space, torch.float32)
        self.replay_buffer.create_storage_space("actions", action_space, torch.float32)
        self.replay_buffer.create_storage_space("log_probs", (), torch.float32)
        self.replay_buffer.create_storage_space("rewards", (), torch.float32)
        self.replay_buffer.create_storage_space("values", (), torch.float32)
        self.replay_buffer.create_storage_space("dones", (), torch.float32)
        
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
        actor_step:Union[StochasticContinuousPolicyStep, DiscretePolicyStep]  = self.actor(obs)
        value_step:ValueStep = self.critic(obs)
        
        action = actor_step.action.tolist()
        log_prob = actor_step.log_prob.tolist()
        value = value_step.value.tolist()
        
        return action, log_prob, value
    
    def rollout(self):
        obs = self.obs
        for i in range(self.rollout_steps):
            self.global_step += self.env_num
            action, log_prob, value = self.get_action(obs)
            next_obs, reward, done, timeout, info = self.envs.step(action)
            
            record = {
                "observations": obs,
                "actions": action,
                "log_probs": log_prob,
                "rewards": reward,
                "values": value,
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
        _, _, value = self.get_action(obs)
        returns, advantages = compute_gae(
            self.replay_buffer.data["rewards"],
            self.replay_buffer.data["values"],
            self.replay_buffer.data["dones"],
            value,
            self.gamma,
            self.lambda_
            )
        
        self.replay_buffer.add_storage("returns", returns)
        self.replay_buffer.add_storage("advantages", advantages)
        
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

                policy_loss, entropy, kl_divergence = PPO.compute_policy_loss(self.actor, log_prob_batch, obs_batch, action_batch, advantage_batch, self.clip_ratio, self.regularization_weight)
 

                value_loss = PPO.compute_clipped_value_loss(self.critic, obs_batch, value_batch, return_batch, self.clip_ratio)
                
                loss = policy_loss + value_loss * self.value_loss_weight - entropy * self.entropy_weight

                self.optimizer.zero_grad()
                loss.backward()
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
            
                
    def train(self, num_epoch:int, num_iteration:int, batch_size:int):
        self.obs, _ = self.envs.reset(seed=[i+self.seed for i in range(self.envs.num_envs)])

        for _ in trange(num_epoch):
            self.rollout()
            self.update(num_iteration, batch_size)
        
        
if __name__ == "__main__":
    trainer = Trainer("HalfCheetah-v5", 20)
    
    trainer.train(num_epoch=100, num_iteration=10, batch_size=500)