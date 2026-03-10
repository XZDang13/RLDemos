from typing import Union, Optional

from tqdm import trange

import torch
import torch.optim as optim

import gymnasium
import numpy as np

from RLAlg.alg.fpo import FPO
from RLAlg.buffer.replay_buffer import ReplayBuffer, compute_gae
from RLAlg.nn.layers import NormPosition
from RLAlg.nn.steps import FPOStep, ValueStep
from RLAlg.utils import set_seed_everywhere
from RLAlg.logger import WandbLogger

from model import Actor, Critic

class Trainer:
    def __init__(self, env_name:str, env_num:int, seed:int=0):
        self.seed = seed
        set_seed_everywhere(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env_name = env_name
        self.env_num = env_num
        self.envs = gymnasium.vector.SyncVectorEnv([lambda: self.setup_env(env_name) for _ in range(env_num)])

        self.max_steps = self.envs.envs[0].spec.max_episode_steps
        self.rollout_steps = 20
        self.flow_steps = 10
        self.sample_per_action = 8

        obs_space = self.envs.single_observation_space.shape
        action_space = self.envs.single_action_space.shape
        
        obs_dim = np.prod(obs_space)
        
        self.max_action = torch.from_numpy(self.envs.single_action_space.high).float().to(self.device)
        action_dim = np.prod(self.envs.single_action_space.shape)
        self.action_dim = action_dim
        self.actor = Actor(obs_dim, action_dim, 32, [128, 128], velocity_scale=0.25, norm_position=NormPosition.POST).to(self.device)

        self.critic = Critic(obs_dim, [128, 128], norm_position=NormPosition.POST).to(self.device)
        
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=3e-4
        )
        
        self.replay_buffer = ReplayBuffer(env_num, self.rollout_steps, device=self.device)
        self.replay_buffer.create_storage_space("observations", obs_space, torch.float32)
        self.replay_buffer.create_storage_space("actions", action_space, torch.float32)
        self.replay_buffer.create_storage_space("eps", (self.sample_per_action, action_dim), torch.float32)
        self.replay_buffer.create_storage_space("time_step", (self.sample_per_action, 1), torch.float32)
        self.replay_buffer.create_storage_space("init_cmf_loss", (self.sample_per_action,), torch.float32)
        self.replay_buffer.create_storage_space("rewards", (), torch.float32)
        self.replay_buffer.create_storage_space("values", (), torch.float32)
        self.replay_buffer.create_storage_space("terminated", (), torch.float32)
        
        self.batch_keys = ["observations", "actions", "eps", "time_step", "init_cmf_loss",  "rewards", "values", "returns", "advantages"]
        
        self.gamma = 0.995
        self.lambda_ = 0.95
        self.clip_ratio = 0.05
        self.max_grad_norm = 1.0
        self.value_loss_weight = 0.25
        
        self.global_step = 0   
        WandbLogger.init_project("RLDemos", f"FPO-{env_name}-{seed}")
        
    def setup_env(self, env_name:str, mode:Optional[str]=None) -> gymnasium.wrappers.RecordEpisodeStatistics:
        env = gymnasium.make(env_name, render_mode=mode)
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)

        return env
    
    @torch.no_grad()
    def get_action(self, obs:np.ndarray):
        obs = torch.from_numpy(obs).float().to(self.device)
        init_noise = torch.randn(self.env_num, self.action_dim).to(self.device)
        fpo_step:FPOStep = FPO.sample_actions_with_cmf_info(self.actor, obs, init_noise, self.flow_steps, False, self.sample_per_action)
        
        value_step:ValueStep = self.critic(obs)
        
        action = fpo_step.action
        eps = fpo_step.eps
        time_step = fpo_step.time_step
        init_cmf_loss = fpo_step.init_cmf_loss
        value = value_step.value
        
        return action, eps, time_step, init_cmf_loss, value
    
    def rollout(self):
        obs = self.obs
        for i in range(self.rollout_steps):
            self.global_step += self.env_num
            action, eps, time_step, init_cmf_loss, value = self.get_action(obs)
            next_obs, reward, terminate, timeout, info = self.envs.step(action.cpu().numpy())
            record = {
                "observations": obs,
                "actions": action,
                "eps": eps,
                "time_step": time_step,
                "init_cmf_loss": init_cmf_loss,
                "rewards": reward,
                "values": value,
                "terminated": terminate
            }
            
            self.replay_buffer.add_records(record)
            
            obs = next_obs
            
            if "episode" in info and "_episode" in info:
                finished = info["_episode"]
                episode_info = {}
                if np.any(finished):
                    episode_info["episode/mean_rewards"] = np.mean(info["episode"]["r"][finished])
                    episode_info["episode/mean_length"] = np.mean(info["episode"]["l"][finished])
                
                    WandbLogger.log_metrics(episode_info, self.global_step)
                
        self.obs = obs
        _, _, _, _, value = self.get_action(obs)
        returns, advantages = compute_gae(
            self.replay_buffer.data["rewards"],
            self.replay_buffer.data["values"],
            self.replay_buffer.data["terminated"],
            value,
            self.gamma,
            self.lambda_
            )
        
        self.replay_buffer.add_storage("returns", returns)
        self.replay_buffer.add_storage("advantages", advantages)
        
    def update(self, num_iteration:int, batch_size:int):
        policy_loss_buffer = []
        value_loss_buffer = []
        cmf_loss_buffer = []
        kl_divergence_buffer = []
        
        for _ in range(num_iteration):
            for batch in self.replay_buffer.sample_batchs(self.batch_keys, batch_size):
                obs_batch = batch["observations"].to(self.device)
                action_batch = batch["actions"].to(self.device)
                eps_batch = batch["eps"].to(self.device)
                time_step_batch = batch["time_step"].to(self.device)
                init_cmf_loss_batch = batch["init_cmf_loss"].to(self.device)
                value_batch = batch["values"].to(self.device)
                return_batch = batch["returns"].to(self.device)
                advantage_batch = batch["advantages"].to(self.device)

                policy_loss_dict = FPO.compute_policy_loss(self.actor, obs_batch, action_batch,
                                                           eps_batch, time_step_batch, init_cmf_loss_batch,
                                                           advantage_batch, self.clip_ratio, average_losses_before_exp=True)

                policy_loss = policy_loss_dict["loss"]
                cmf_loss = policy_loss_dict["cmf_loss"]
                kl_divergence = policy_loss_dict["kl_divergence"]

                value_loss_dict = FPO.compute_clipped_value_loss(self.critic, obs_batch, value_batch, return_batch, self.clip_ratio)
                
                value_loss = value_loss_dict["loss"]

                loss = policy_loss + value_loss * self.value_loss_weight

                self.optimizer.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                policy_loss_buffer.append(policy_loss.item())
                value_loss_buffer.append(value_loss.item())
                cmf_loss_buffer.append(cmf_loss.item())
                kl_divergence_buffer.append(kl_divergence.item())
                
        avg_policy_loss = np.mean(policy_loss_buffer)
        avg_value_loss = np.mean(value_loss_buffer)
        avg_cmf_loss = np.mean(cmf_loss_buffer)
        avg_kl_divergence = np.mean(kl_divergence_buffer)
        
        train_info = {
            "update/avg_policy_loss": avg_policy_loss,
            "update/avg_value_loss": avg_value_loss,
            "update/avg_cmf_loss": avg_cmf_loss,
            "update/avg_kl_divergence": avg_kl_divergence
        }

        WandbLogger.log_metrics(train_info, self.global_step)
            
                
    def train(self, num_epoch:int, num_iteration:int, batch_size:int):
        self.obs, _ = self.envs.reset(seed=[i+self.seed for i in range(self.envs.num_envs)])

        for _ in trange(num_epoch):
            self.rollout()
            self.update(num_iteration, batch_size)
        
        
if __name__ == "__main__":
    trainer = Trainer("HalfCheetah-v5", 50, seed=100)
    
    trainer.train(num_epoch=5000, num_iteration=10, batch_size=500)
