"""Train IQL only from a fixed dataset; environment interaction is optional evaluation."""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from RLAlg.alg.iql import IQL
from RLAlg.utils import set_seed_everywhere

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from IQL.model import Actor, Critic, Value
from demo_utils import (
    DemoLogger, continuous_action_limit, default_checkpoint_path, mean_metrics, save_checkpoint, setup_env,
)


class Trainer:
    def __init__(self, env_name, dataset, seed=0, *, device=None, use_wandb=False,
                 expectile=0.7, temperature=3.0):
        if not 0 <= expectile <= 1 or temperature <= 0:
            raise ValueError("expectile must be in [0, 1] and temperature must be positive.")
        self.seed = seed
        set_seed_everywhere(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.env_name = env_name
        self.env = setup_env(env_name)
        self.max_action = continuous_action_limit(self.env.action_space, self.device)
        self.dataset = self.load_dataset(dataset)
        self.dataset_size = len(self.dataset["rewards"])
        obs_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = self.max_action.numel()
        self.actor = Actor(obs_dim, action_dim, [128, 128], self.max_action).to(self.device)
        self.critic = Critic(obs_dim, action_dim, [128, 128]).to(self.device)
        self.value = Value(obs_dim, [128, 128]).to(self.device)
        self.critic_target = Critic(obs_dim, action_dim, [128, 128]).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=3e-4)
        self.expectile = expectile
        self.temperature = temperature
        self.gamma = 0.99
        self.tau = 0.005
        self.max_grad_norm = 1.0
        self.global_step = 0
        self.logger = DemoLogger(f"IQL-{env_name}-{seed}", use_wandb)

    def load_dataset(self, dataset):
        if isinstance(dataset, (str, Path)):
            with np.load(dataset, allow_pickle=False) as file:
                dataset = {key: file[key] for key in file.files}
        if "env_name" in dataset and str(np.asarray(dataset["env_name"]).item()) != self.env_name:
            raise ValueError("Dataset env_name does not match the requested environment.")
        required = {"observations", "actions", "rewards", "next_observations", "terminated"}
        missing = required - dataset.keys()
        if missing:
            raise ValueError(f"Dataset is missing keys: {sorted(missing)}")
        rewards = np.asarray(dataset["rewards"])
        if rewards.ndim != 1 or not len(rewards):
            raise ValueError("Dataset rewards must have non-empty shape [transitions].")
        size = len(rewards)
        shapes = {
            "observations": (size, *self.env.observation_space.shape),
            "next_observations": (size, *self.env.observation_space.shape),
            "actions": (size, *self.env.action_space.shape),
            "rewards": (size,), "terminated": (size,), "truncated": (size,),
        }
        result = {}
        for key, shape in shapes.items():
            values = np.asarray(dataset.get(key, np.zeros(size, dtype=bool)))
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"Dataset {key} must be finite with shape {shape}.")
            is_mask = key in {"terminated", "truncated"}
            if is_mask and not np.isin(values, [0, 1]).all():
                raise ValueError(f"Dataset {key} must contain only boolean or 0/1 values.")
            result[key] = torch.as_tensor(values, dtype=torch.bool if is_mask else torch.float32)
        limit = self.max_action.cpu()
        if torch.any(result["actions"].abs() > limit + 1e-6):
            raise ValueError("Dataset actions exceed the environment action bounds.")
        return result

    def sample_batch(self, batch_size):
        indices = torch.randint(self.dataset_size, (batch_size,))
        return {key: value[indices].to(self.device) for key, value in self.dataset.items()}

    def optimize(self, result, model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        optimizer.step()

    def update(self, num_iteration, batch_size):
        metrics = defaultdict(list)
        for _ in range(num_iteration):
            batch = self.sample_batch(batch_size)
            value = IQL.compute_value_loss(
                self.value, self.critic_target,
                observations=batch["observations"], actions=batch["actions"], expectile=self.expectile,
            )
            self.optimize(value, self.value, self.value_optimizer)
            critic = IQL.compute_critic_loss(
                self.value, self.critic,
                observations=batch["observations"], actions=batch["actions"], rewards=batch["rewards"],
                next_observations=batch["next_observations"], terminated=batch["terminated"], gamma=self.gamma,
            )
            self.optimize(critic, self.critic, self.critic_optimizer)
            policy = IQL.compute_policy_loss(
                self.actor, self.value, self.critic_target,
                observations=batch["observations"], actions=batch["actions"], temperature=self.temperature,
            )
            self.optimize(policy, self.actor, self.actor_optimizer)
            IQL.update_target_param(self.critic, self.critic_target, self.tau)
            self.global_step += 1
            for prefix, result in (("value", value), ("critic", critic), ("policy", policy)):
                for key, metric in result.items():
                    name = f"{prefix}_loss" if key == "loss" else key
                    metrics[f"update/avg_{name}"].append(metric.detach().item())
        train_info = mean_metrics(metrics)
        self.logger.log_metrics(train_info, self.global_step)
        return train_info

    def train(self, num_epoch, num_iteration, batch_size):
        for _ in trange(num_epoch):
            self.update(num_iteration, batch_size)

    @torch.no_grad()
    def evaluate(self, episodes=5):
        returns = []
        for episode in range(episodes):
            obs, _ = self.env.reset(seed=self.seed + episode)
            total_reward = 0.0
            while True:
                observations = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                action = self.actor(observations).mean.squeeze(0).cpu().numpy()
                obs, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            returns.append(total_reward)
        metrics = {"evaluation/mean_rewards": float(np.mean(returns))}
        self.logger.log_metrics(metrics, self.global_step)
        return metrics

    def close(self):
        self.env.close()
        self.logger.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="Pendulum-v1")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--save-path", default=None,
                        help="Checkpoint path. Defaults to checkpoints/IQL-<env>-<seed>.pt.")
    args = parser.parse_args()
    if min(args.epochs, args.iterations, args.batch_size) <= 0 or args.eval_episodes < 0:
        parser.error("Training counts must be positive and --eval-episodes must be non-negative.")
    if not args.dataset.is_file():
        parser.error("Dataset does not exist. Supply a dataset or create one with IQL.collect.")
    trainer = Trainer(
        args.env, args.dataset, args.seed, device=args.device, use_wandb=args.wandb,
        expectile=args.expectile, temperature=args.temperature,
    )
    try:
        trainer.train(args.epochs, args.iterations, args.batch_size)
        save_checkpoint(
            trainer, args.save_path or default_checkpoint_path("IQL", args.env, args.seed),
            algorithm="IQL", env_name=args.env, seed=args.seed,
        )
        if args.eval_episodes:
            trainer.evaluate(args.eval_episodes)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
