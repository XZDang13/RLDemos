from typing import Union, Optional

from tqdm import trange

import torch
import torch.optim as optim

import gymnasium
import numpy as np

from RLAlg.alg.ppo import PPO
from RLAlg.alg.gan import GAN
from RLAlg.buffer.replay_buffer import ReplayBuffer, compute_gae
from RLAlg.nn.steps import StochasticContinuousPolicyStep, DiscretePolicyStep, ValueStep
from RLAlg.utils import set_seed_everywhere
from RLAlg.logger import WandbLogger

from model import ContinuousActor, DiscreteActor, Critic, Discriminator

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

        obs_space = self.envs.single_observation_space.shape
        action_space = self.envs.single_action_space.shape
        
        obs_dim = np.prod(obs_space)
        if isinstance(self.envs.single_action_space, gymnasium.spaces.Discrete):
            action_dim = self.envs.single_action_space.n
            self.actor = DiscreteActor(obs_dim, action_dim, [128, 128]).to(self.device)
        elif isinstance(self.envs.single_action_space, gymnasium.spaces.Box):
            self.max_action = torch.from_numpy(self.envs.single_action_space.high).float().to(self.device)
            action_dim = np.prod(self.envs.single_action_space.shape)
            self.actor = ContinuousActor(obs_dim, action_dim, [128, 128], self.max_action).to(self.device)

        self.critic = Critic(obs_dim, [128, 128]).to(self.device)
        self.discriminator = Discriminator(obs_dim+action_dim, [128, 128]).to(self.device)
        
        self.ac_optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=3e-4
        )

        self.d_optimizer = torch.optim.Adam(
            [
                {'params': self.discriminator.layers.parameters(), "weight_decay":1e-4},
                {'params': self.discriminator.head.parameters(), "weight_decay":1e-2},
            ],
            lr=1e-5, betas=(0.5, 0.999)
        )
        
        self.replay_buffer = ReplayBuffer(env_num, self.max_steps, device=self.device)
        self.replay_buffer.create_storage_space("observations", obs_space, torch.float32)
        self.replay_buffer.create_storage_space("actions", action_space, torch.float32)
        self.replay_buffer.create_storage_space("log_probs", (), torch.float32)
        self.replay_buffer.create_storage_space("rewards", (), torch.float32)
        self.replay_buffer.create_storage_space("values", (), torch.float32)
        self.replay_buffer.create_storage_space("dones", (), torch.float32)

        self.expert_buffer = ReplayBuffer(env_num, 1000, device=self.device)
        self.expert_buffer.load("GAIL/HalfCheetah-v5_expert_data.pth", device=self.device)
        
        self.batch_keys = ["observations", "actions", "log_probs", "rewards", "values", "returns", "advantages"]
        
        self.gamma = 0.99
        self.lambda_ = 0.95
        self.clip_ratio = 0.2
        self.regularization_weight = 0.0
        self.max_grad_norm = 1.0
        self.value_loss_weight = 0.5
        self.entropy_weight = 0.01
        
        self.global_step = 0   
        WandbLogger.init_project("RLDemos", f"GAIL-{env_name}-{seed}")
        
    def setup_env(self, env_name:str, mode:Optional[str]=None) -> gymnasium.wrappers.RecordEpisodeStatistics:
        env = gymnasium.make(env_name, render_mode=mode)
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)

        return env
    
    @torch.no_grad()
    def get_action(self, obs:np.ndarray):
        obs = torch.from_numpy(obs).float().to(self.device)
        actor_step:Union[StochasticContinuousPolicyStep, DiscretePolicyStep]  = self.actor(obs)
        value_step:ValueStep = self.critic(obs)

        d_obs = torch.cat([obs, actor_step.action], dim=1)
        style_reward_step:ValueStep = self.discriminator(d_obs)
        
        action = actor_step.action.tolist()
        log_prob = actor_step.log_prob.tolist()
        value = value_step.value.tolist()
        style_reward = (-torch.log(1-1/(1+torch.exp(-style_reward_step.value))+1e-5)).tolist()
        
        return action, log_prob, value, style_reward
    
    
    def rollout(self):
        obs = self.obs
        for i in range(self.rollout_steps):
            self.global_step += self.env_num
            action, log_prob, value, style_reward= self.get_action(obs)
            next_obs, task_reward, done, timeout, info = self.envs.step(action)
            
            record = {
                "observations": obs,
                "actions": action,
                "log_probs": log_prob,
                "rewards": style_reward,
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
        _, _, value, _ = self.get_action(obs)
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
        discriminator_loss_buffer = []
        
        for _ in range(num_iteration):
            for batch in self.replay_buffer.sample_batchs(self.batch_keys, batch_size):
                obs_batch = batch["observations"].to(self.device)
                action_batch = batch["actions"].to(self.device)
                log_prob_batch = batch["log_probs"].to(self.device)
                value_batch = batch["values"].to(self.device)
                return_batch = batch["returns"].to(self.device)
                advantage_batch = batch["advantages"].to(self.device)

                policy_loss_dict = PPO.compute_policy_loss(self.actor, log_prob_batch, obs_batch, action_batch, advantage_batch, self.clip_ratio, self.regularization_weight)

                policy_loss = policy_loss_dict["loss"]
                entropy = policy_loss_dict["entropy"]
                kl_divergence = policy_loss_dict["kl_divergence"]

                value_loss_dict = PPO.compute_clipped_value_loss(self.critic, obs_batch, value_batch, return_batch, self.clip_ratio)
                
                value_loss = value_loss_dict["loss"]

                ac_loss = policy_loss + value_loss * self.value_loss_weight - entropy * self.entropy_weight

                self.ac_optimizer.zero_grad()
                ac_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.ac_optimizer.step()

                expert_batch = self.expert_buffer.sample_batch(["observations", "actions"], batch_size)
                expert_obs_batch = expert_batch["observations"].to(self.device)
                expert_action_batch = expert_batch["actions"].to(self.device)

                expert_d_obs_batch = torch.cat([expert_obs_batch, expert_action_batch], dim=1)
                agent_d_obs_batch = torch.cat([obs_batch, action_batch], dim=1)

                d_loss_dict = GAN.compute_bce_loss(self.discriminator, expert_d_obs_batch, agent_d_obs_batch, r1_gamma=5.0)
                
                d_loss = d_loss_dict["loss"]
                d_loss_real = d_loss_dict["loss_real"]
                d_loss_fake = d_loss_dict["loss_fake"]
                d_loss_gp = d_loss_dict["gradient_penalty"]
                
                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()
                
                policy_loss_buffer.append(policy_loss.item())
                value_loss_buffer.append(value_loss.item())
                entropy_buffer.append(entropy.item())
                kl_divergence_buffer.append(kl_divergence.item())
                discriminator_loss_buffer.append(d_loss.item())
                
        avg_policy_loss = np.mean(policy_loss_buffer)
        avg_value_loss = np.mean(value_loss_buffer)
        avg_entropy = np.mean(entropy_buffer)
        avg_kl_divergence = np.mean(kl_divergence_buffer)
        avg_discriminator_loss = np.mean(discriminator_loss_buffer)
        
        train_info = {
            "update/avg_policy_loss": avg_policy_loss,
            "update/avg_value_loss": avg_value_loss,
            "update/avg_entropy": avg_entropy,
            "update/avg_kl_divergence": avg_kl_divergence,
            "update/avg_discriminator_loss": avg_discriminator_loss
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
