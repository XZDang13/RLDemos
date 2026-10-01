import sys
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from tqdm import trange

from RLAlg.alg.ppo import PPO
from RLAlg.buffer.replay_buffer import ReplayBuffer, compute_gae
from RLAlg.utils import set_seed_everywhere

if __package__:
    from .model import ContinuousActor, DiscreteActor, Critic
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model import ContinuousActor, DiscreteActor, Critic

from demo_utils import (
    DemoLogger, continuous_action_limit, episode_metrics, make_vector_env,
    mean_metrics, reset_finished_envs, run_training, setup_env,
)


class Trainer:
    algorithm_name = "PPO"

    def __init__(self, env_name, env_num, seed=0, *, rollout_steps=None,
                 device=None, use_wandb=False, std_parameterization="log_std",
                 init_std=1.0, bootstrap_timeouts=True):
        self.seed = seed
        set_seed_everywhere(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.env_name = env_name
        self.env_num = env_num
        self.envs = make_vector_env(env_name, env_num)
        self.max_steps = self.envs.envs[0].spec.max_episode_steps
        self.rollout_steps = rollout_steps if rollout_steps is not None else self.max_steps
        if not self.rollout_steps or self.rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive.")
        if self.rollout_steps * env_num <= 1:
            raise ValueError("GAE normalization requires at least two rollout transitions.")
        self.bootstrap_timeouts = bootstrap_timeouts
        obs_space = self.envs.single_observation_space.shape
        action_space = self.envs.single_action_space
        obs_dim = int(np.prod(obs_space))
        self.actor, action_dtype = self.create_actor(obs_dim, action_space, std_parameterization, init_std)
        self.critic = Critic(obs_dim, [128, 128]).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=3e-4,
        )
        self.replay_buffer = ReplayBuffer(env_num, self.rollout_steps, device=self.device)
        for key, shape, dtype in (
            ("observations", obs_space, torch.float32),
            ("actions", action_space.shape, action_dtype),
            ("log_probs", (), torch.float32),
            ("rewards", (), torch.float32),
            ("values", (), torch.float32),
            ("terminated", (), torch.bool),
            ("truncated", (), torch.bool),
        ):
            self.replay_buffer.create_storage_space(key, shape, dtype)
        self.batch_keys = ["observations", "actions", "log_probs", "values", "returns", "advantages"]
        self.gamma = 0.99
        self.lambda_ = 0.95
        self.clip_ratio = 0.2
        self.regularization_weight = 0.0
        self.max_grad_norm = 1.0
        self.value_loss_weight = 0.5
        self.entropy_weight = 0.01
        self.global_step = 0
        self.logger = DemoLogger(f"{self.algorithm_name}-{env_name}-{seed}", use_wandb)

    setup_env = staticmethod(setup_env)

    def create_actor(self, obs_dim, action_space, std_parameterization, init_std):
        if isinstance(action_space, gym.spaces.Discrete):
            self.actor = DiscreteActor(obs_dim, action_space.n, [128, 128]).to(self.device)
            action_dtype = torch.long
        elif isinstance(action_space, gym.spaces.Box):
            self.max_action = continuous_action_limit(action_space, self.device)
            self.actor = ContinuousActor(
                obs_dim, int(np.prod(action_space.shape)), [128, 128], self.max_action,
                std_parameterization=std_parameterization, init_std=init_std,
            ).to(self.device)
            action_dtype = torch.float32
        else:
            raise ValueError("PPO supports Discrete or one-dimensional Box actions.")
        return self.actor, action_dtype

    @torch.no_grad()
    def get_action(self, obs):
        observations = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actor_step = self.actor(observations)
        value = self.critic(observations).value
        return actor_step.action.cpu().numpy(), actor_step.log_prob, value

    @torch.no_grad()
    def timeout_rewards(self, rewards, next_obs, terminated, truncated):
        rewards = np.array(rewards, copy=True)
        timeouts = truncated & ~terminated
        if self.bootstrap_timeouts and timeouts.any():
            observations = torch.as_tensor(next_obs[timeouts], dtype=torch.float32, device=self.device)
            # Gymnasium supplies the actual final observation before manual reset.
            rewards[timeouts] += self.gamma * self.critic(observations).value.cpu().numpy()
        return rewards

    @torch.no_grad()
    def finish_rollout(self, obs):
        observations = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        last_values = self.critic(observations).value
        data = self.replay_buffer.data
        returns, advantages = compute_gae(
            data["rewards"], data["values"], data["terminated"], last_values,
            self.gamma, self.lambda_, truncated=data["truncated"],
            # Timeout rewards already bootstrap from V(final_obs). Avoid adding
            # the library's optional V(current_obs) timeout correction as well.
            bootstrap_timeouts=False,
        )
        self.replay_buffer.add_storage("returns", returns)
        self.replay_buffer.add_storage("advantages", advantages)

    def rollout(self):
        self.replay_buffer.reset()
        obs = self.obs
        for _ in range(self.rollout_steps):
            action, log_prob, value = self.get_action(obs)
            next_obs, reward, terminated, truncated, info = self.envs.step(action)
            self.global_step += self.env_num
            self.replay_buffer.add_records({
                "observations": obs,
                "actions": action,
                "log_probs": log_prob,
                "rewards": self.timeout_rewards(reward, next_obs, terminated, truncated),
                "values": value,
                "terminated": terminated,
                "truncated": truncated,
            })
            self.logger.log_metrics(episode_metrics(info), self.global_step)
            obs = reset_finished_envs(self.envs, next_obs, terminated, truncated)
        self.obs = obs
        self.finish_rollout(obs)

    def update(self, num_iteration, batch_size):
        metrics = defaultdict(list)
        for _ in range(num_iteration):
            for batch in self.replay_buffer.sample_batchs(self.batch_keys, batch_size):
                policy = PPO.compute_policy_loss(
                    self.actor, batch["log_probs"], batch["observations"], batch["actions"],
                    batch["advantages"], self.clip_ratio,
                    entropy_coef=self.entropy_weight,
                    regularization_weight=self.regularization_weight,
                )
                value = PPO.compute_clipped_value_loss(
                    self.critic, batch["observations"], batch["values"], batch["returns"], self.clip_ratio,
                )
                loss = policy["loss"] + self.value_loss_weight * value["loss"]
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                for prefix, result in (("policy", policy), ("value", value)):
                    for key, metric in result.items():
                        name = f"{prefix}_loss" if key == "loss" else key
                        metrics[f"update/avg_{name}"].append(metric.detach().item())
        train_info = mean_metrics(metrics)
        self.logger.log_metrics(train_info, self.global_step)
        return train_info

    def train(self, num_epoch, num_iteration, batch_size):
        self.obs, _ = self.envs.reset(seed=self.seed)
        for _ in trange(num_epoch):
            self.rollout()
            self.update(num_iteration, batch_size)

    def close(self):
        self.envs.close()
        self.logger.finish()


if __name__ == "__main__":
    run_training(Trainer, "PPO", default_iterations=10)
