from typing import Union, Optional

from tqdm import trange

import torch
import torch.optim as optim

import gymnasium
import numpy as np

from RLAlg.alg.ddpg import DDPG
from RLAlg.buffer.replay_buffer import ReplayBuffer
from RLAlg.nn.steps import DeterministicContinuousPolicyStep, ValueStep
from RLAlg.utils import set_seed_everywhere

from model import Actor, Critic

class Trainer:
    def __init__(self, env_name:str, env_num:int, seed:int=0):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        
        set_seed_everywhere(self.seed)

        self.env_name = env_name
        self.envs = gymnasium.vector.SyncVectorEnv([lambda: self.setup_env(env_name) for _ in range(env_num)])

        self.max_steps = self.envs.envs[0].spec.max_episode_steps
        self.rollout_steps = self.max_steps
        self.max_buffer_steps = 100000

        self.max_action = torch.from_numpy(self.envs.single_action_space.high).float()
        obs_space = self.envs.single_observation_space.shape
        action_space = self.envs.single_action_space.shape

        self.actor = Actor(np.prod(obs_space), np.prod(action_space), [128, 128], self.max_action)
        self.critic = Critic(np.prod(obs_space), np.prod(action_space), [128, 128])
        self.actor_target = Actor(np.prod(obs_space), np.prod(action_space), [128, 128], self.max_action)
        self.critic_target = Critic(np.prod(obs_space), np.prod(action_space), [128, 128])
        
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
        self.alpha = 0.2
        self.regularization_weight = 0.0
        self.tau = 0.005
        self.max_grad_norm = 1.0
        
        
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
            
        return action.tolist()
    
    def average_non_zero(self, numbers):
        non_zero_numbers = [num for num in numbers if num != 0]
        if not non_zero_numbers:
            return 0  # Return 0 if there are no non-zero elements
        return sum(non_zero_numbers) / len(non_zero_numbers)
    
    def rollout(self, random:bool=False):
        obs = self.obs
        for i in range(self.rollout_steps):
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
                print(self.average_non_zero(info['episode']['r']))
        
        self.obs = obs
        
    def update(self, num_iteration:int, batch_size:int):
        for _ in range(num_iteration):
            batch = self.replay_buffer.sample_batch(self.batch_keys, batch_size)
            obs_batch = batch["observations"].to(self.device)
            next_obs_batch = batch["next_observations"].to(self.device)
            action_batch = batch["actions"].to(self.device)
            reward_batch = batch["rewards"].to(self.device)
            done_batch = batch["dones"].to(self.device)
            
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss = DDPG.compute_critic_loss(self.actor_target, self.critic, self.critic_target,
                                                   obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, self.gamma)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            for param in self.critic.parameters():
                param.requires_grad = False

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss = DDPG.compute_actor_loss(self.actor, self.critic, obs_batch.detach(), self.regularization_weight)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            for param in self.critic.parameters():
                param.requires_grad = True

            DDPG.update_target_param(self.actor, self.actor_target, self.tau)
            DDPG.update_target_param(self.critic, self.critic_target, self.tau)
                
    def train(self, num_epoch:int, num_iteration:int, batch_size:int):
        self.obs, _ = self.envs.reset(seed=[i+self.seed for i in range(self.envs.num_envs)])
        random = True
        for i in trange(num_epoch):
            if i > (num_epoch // 10):
                random = False
            self.rollout(random)
            self.update(num_iteration, batch_size)
        
        
if __name__ == "__main__":
    trainer = Trainer("HalfCheetah-v5", 20)
    
    trainer.train(num_epoch=100, num_iteration=250, batch_size=500)