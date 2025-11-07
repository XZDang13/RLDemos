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
        self.rollout_steps = self.max_steps
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
        self.critic_target = Critic(obs_dim, action_dim, [128, 128]).to(self.device)
        
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
        actor_step:Union[StochasticContinuousPolicyStep, DiscretePolicyStep]  = self.actor(obs)
        
        if random:
            action = actor_step.action.uniform_(-1, 1) * self.max_action
        else:
            action = actor_step.action
            
        return action.tolist()
    
    def rollout(self, random:bool=False):
        obs = self.obs
        for i in range(self.rollout_steps):
            self.global_step += self.env_num
            
            action = self.get_action(obs, random)
            next_obs, reward, done, timeout, info = self.envs.step(action)
            
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
            critic_loss_dict = SAC.compute_critic_loss(self.actor, self.critic, self.critic_target, obs_batch, action_batch, reward_batch,
                                                  next_obs_batch, done_batch, self.alpha, self.gamma)
            
            critic_loss = critic_loss_dict["loss"]
            q1 = critic_loss_dict["q1"]
            q2 = critic_loss_dict["q2"]
            q_target = critic_loss_dict["q_target"]

            critic_loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            for param in self.critic.parameters():
                param.requires_grad = False

            self.actor_optimizer.zero_grad(set_to_none=True)
            policy_loss_dict = SAC.compute_policy_loss(self.actor, self.critic, obs_batch.detach(), self.alpha, self.regularization_weight)
            
            policy_loss = policy_loss_dict["loss"]
            
            policy_loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            for param in self.critic.parameters():
                param.requires_grad = True

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
                
    def train(self, num_epoch:int, num_iteration:int, batch_size:int):
        self.obs, _ = self.envs.reset(seed=[i+self.seed for i in range(self.envs.num_envs)])
        random = True
        for i in trange(num_epoch):
            if i > (num_epoch // 5):
                random = False
            self.rollout(random)
            self.update(num_iteration, batch_size)
        
        
if __name__ == "__main__":
    trainer = Trainer("HalfCheetah-v5", 20, seed=100)
    
    trainer.train(num_epoch=100, num_iteration=250, batch_size=500)
    