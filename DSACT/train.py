import sys
from pathlib import Path

from tqdm import trange

import torch
import torch.optim as optim

import numpy as np

from RLAlg.alg.dsact import DSACT
from RLAlg.buffer.replay_buffer import ReplayBuffer
from RLAlg.utils import set_seed_everywhere

if __package__:
    from .model import Actor, Critic
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model import Actor, Critic

from demo_utils import (
    DemoLogger, continuous_action_limit, episode_metrics, make_vector_env,
    reset_finished_envs, run_training, setup_env,
)


class Trainer:
    def __init__(self, env_name:str, env_num:int, seed:int=0, *,
                 rollout_steps:int | None=None, buffer_steps:int=100000,
                 device:str | None=None, use_wandb:bool=False):
        self.seed = seed
        set_seed_everywhere(self.seed)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.env_name = env_name
        self.env_num = env_num
        self.envs = make_vector_env(env_name, env_num)

        self.max_steps = self.envs.envs[0].spec.max_episode_steps
        self.rollout_steps = rollout_steps if rollout_steps is not None else self.max_steps
        self.max_buffer_steps = buffer_steps
        if not self.rollout_steps or self.rollout_steps <= 0 or buffer_steps <= 0:
            raise ValueError("rollout_steps and buffer_steps must be positive.")
        self.envs.action_space.seed(seed)

        self.max_action = continuous_action_limit(self.envs.single_action_space, self.device)
        obs_space = self.envs.single_observation_space.shape
        action_space = self.envs.single_action_space.shape

        obs_dim = int(np.prod(obs_space))
        action_dim = int(np.prod(action_space))

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
        self.replay_buffer.create_storage_space("dones", (), torch.bool)

        self.batch_keys = ["observations", "next_observations", "actions", "rewards", "dones"]

        self.gamma = 0.99
        self.alpha = 0.2
        self.regularization_weight = 0.0
        self.tau = 0.005
        # Running means of the q1/q2 std, owned by this trainer; None until the first critic update.
        # Running means of the critic stds, updated by every critic loss and saved in checkpoints.
        self.q_mean_std = None
        self.extra_checkpoint_state = ("q_mean_std",)
        self.tau_b = 0.005
        self.max_grad_norm = 1.0

        self.global_step = 0
        self.logger = DemoLogger(f"DSACT-{env_name}-{seed}", use_wandb)

    setup_env = staticmethod(setup_env)

    @torch.no_grad()
    def get_action(self, obs:np.ndarray, random:bool=False):
        if random:
            return self.envs.action_space.sample()
        observations = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        step = self.actor(observations)
        action = step.action
        action = torch.clamp(action, -self.max_action, self.max_action)
        return action.cpu().numpy()

    def rollout(self, random:bool=False):
        obs = self.obs
        for _ in range(self.rollout_steps):
            self.global_step += self.env_num

            action = self.get_action(obs, random)
            next_obs, reward, done, timeout, info = self.envs.step(action)

            record = {
                "observations": obs,
                "next_observations": next_obs,
                "actions": action,
                "rewards": reward,
                "dones": done,  # True termination only; timeouts still bootstrap.
            }

            self.replay_buffer.add_records(record)

            obs = reset_finished_envs(self.envs, next_obs, done, timeout)

            self.logger.log_metrics(episode_metrics(info), self.global_step)

        self.obs = obs

    def update(self, num_iteration:int, batch_size:int):
        policy_loss_buffer = []
        critic_loss_buffer = []
        critic_metrics = {}

        for _ in range(num_iteration):
            batch = self.replay_buffer.sample_batch(self.batch_keys, batch_size)
            obs_batch = batch["observations"].to(self.device)
            next_obs_batch = batch["next_observations"].to(self.device)
            action_batch = batch["actions"].to(self.device)
            reward_batch = batch["rewards"].to(self.device)
            done_batch = batch["dones"].to(self.device)

            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_result = DSACT.compute_critic_loss(
                self.actor, self.critic, self.critic_target,
                observations=obs_batch, actions=action_batch, rewards=reward_batch,
                next_observations=next_obs_batch, terminated=done_batch, alpha=self.alpha,
                gamma=self.gamma, tau_b=self.tau_b, q_mean_std=self.q_mean_std,
            )
            self.q_mean_std = critic_result.pop("q_mean_std")
            critic_loss = critic_result["loss"]
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            for param in self.critic.parameters():
                param.requires_grad = False

            self.actor_optimizer.zero_grad(set_to_none=True)
            policy_result = DSACT.compute_policy_loss(
                self.actor, self.critic, observations=obs_batch.detach(), alpha=self.alpha,
                regularization_weight=self.regularization_weight,
            )
            policy_loss = policy_result["loss"]
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            for param in self.critic.parameters():
                param.requires_grad = True

            DSACT.update_target_param(self.critic, self.critic_target, self.tau)

            policy_loss_buffer.append(policy_loss.item())
            critic_loss_buffer.append(critic_loss.item())
            for key, value in critic_result.items():
                if key != "loss":
                    critic_metrics.setdefault(key, []).append(value.detach().item())

        avg_policy_loss = np.mean(policy_loss_buffer)
        avg_critic_loss = np.mean(critic_loss_buffer)

        train_info = {
            "update/avg_policy_loss": avg_policy_loss,
            "update/avg_critic_loss": avg_critic_loss,
            **{f"update/avg_{key}": float(np.mean(values)) for key, values in critic_metrics.items()},
        }

        self.logger.log_metrics(train_info, self.global_step)
        return train_info

    def train(self, num_epoch:int, num_iteration:int, batch_size:int):
        self.obs, _ = self.envs.reset(seed=[i+self.seed for i in range(self.envs.num_envs)])
        for i in trange(num_epoch):
            self.rollout(random=i < max(1, num_epoch // 10))
            self.update(num_iteration, batch_size)

    def close(self):
        self.envs.close()
        self.logger.finish()


if __name__ == "__main__":
    run_training(Trainer, "DSACT")
