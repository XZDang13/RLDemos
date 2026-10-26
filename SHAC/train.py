import argparse
import copy
import importlib
import sys
from pathlib import Path

from tqdm import trange

import torch
import torch.optim as optim

import numpy as np

from RLAlg.alg.shac import SHAC
from RLAlg.normalizer import Normalizer
from RLAlg.utils import set_seed_everywhere

if __package__:
    from .model import Actor, Critic
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model import Actor, Critic

from demo_utils import DemoLogger, default_checkpoint_path, save_checkpoint


REWARPED_ENVS = {
    "Ant": "rewarped.envs.dflex.ant",
    "Hopper": "rewarped.envs.dflex.hopper",
    "Humanoid": "rewarped.envs.dflex.humanoid",
    "Cartpole": "rewarped.envs.dflex.cartpole",
    "Anymal": "rewarped.envs.dflex.anymal",
    "A1": "rewarped.envs.dflex.a1",
}


def make_rewarped_env(env_name:str, env_num:int, seed:int, device:torch.device):
    if env_name not in REWARPED_ENVS:
        raise ValueError(f"env_name must be one of {sorted(REWARPED_ENVS)}, got {env_name!r}.")
    # Imported lazily so the demo tests can run without Rewarped installed.
    env_type = getattr(importlib.import_module(REWARPED_ENVS[env_name]), env_name)
    env = env_type(num_envs=env_num, seed=seed, no_grad=False, device=str(device))
    env.init()
    env.reset()
    return env


class Trainer:
    """SHAC on a differentiable simulator with the Rewarped/DFlex env API.

    ``env.initialize_trajectory()`` cuts the graph and returns observations;
    ``env.step(actions)`` returns ``(obs, reward, done, extras)`` where extras
    holds ``terminated``, ``truncated`` and ``obs_before_reset``.
    """

    def __init__(self, env_name:str="Ant", env_num:int=128, seed:int=0, *,
                 horizon:int=32, max_epochs:int=2000, device:str | None=None,
                 use_wandb:bool=False, env=None):
        self.seed = seed
        set_seed_everywhere(self.seed)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if env_num <= 0 or horizon <= 0 or max_epochs <= 0:
            raise ValueError("env_num, horizon and max_epochs must be positive.")
        self.env_name = env_name
        self.env = env if env is not None else make_rewarped_env(env_name, env_num, seed, self.device)
        self.env_num = self.env.num_envs
        self.horizon = horizon
        self.max_epochs = max_epochs

        obs_dim = self.env.num_obs
        action_dim = self.env.num_act

        # Network sizes, learning rates and target rate follow the reference SHAC Ant config.
        self.actor = Actor(obs_dim, action_dim, [400, 200, 100]).to(self.device)
        self.critic = Critic(obs_dim, [400, 200, 100]).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.actor_lr = 2e-3
        self.critic_lr = 5e-4
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr, betas=(0.7, 0.95))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr, betas=(0.7, 0.95))

        self.obs_normalizer = Normalizer(obs_dim).to(self.device)

        self.gamma = 0.99
        self.lambda_ = 0.95
        self.tau = 0.01
        self.critic_iterations = 16
        self.critic_batches = 4
        self.max_grad_norm = 1.0

        self.global_step = 0
        self.episode_return = torch.zeros(self.env_num, device=self.device)
        self.episode_length = torch.zeros(self.env_num, device=self.device)
        self.logger = DemoLogger(f"SHAC-{env_name}-{seed}", use_wandb)

    def set_learning_rates(self, epoch:int):
        # Linear decay to 1e-5, as in the reference implementation.
        progress = epoch / self.max_epochs
        for optimizer, lr in ((self.actor_optimizer, self.actor_lr), (self.critic_optimizer, self.critic_lr)):
            for group in optimizer.param_groups:
                group["lr"] = (1e-5 - lr) * progress + lr

    def track_episodes(self, reward:torch.Tensor, done:torch.Tensor, finished_returns:list, finished_lengths:list):
        self.episode_return += reward.detach()
        self.episode_length += 1
        if done.any():
            finished_returns.extend(self.episode_return[done].tolist())
            finished_lengths.extend(self.episode_length[done].tolist())
            self.episode_return[done] = 0.0
            self.episode_length[done] = 0.0

    def rollout(self):
        # Freeze the statistics for this rollout so all steps share one normalization.
        normalizer = copy.deepcopy(self.obs_normalizer)
        raw_obs = self.env.initialize_trajectory()
        obs = normalizer(raw_obs)

        raw_obs_buffer, obs_buffer, rewards, next_values, terminated, truncated = [], [], [], [], [], []
        finished_returns, finished_lengths = [], []
        for _ in range(self.horizon):
            self.global_step += self.env_num
            raw_obs_buffer.append(raw_obs.detach())
            obs_buffer.append(obs.detach())

            step = self.actor(obs)
            raw_obs, reward, done, extras = self.env.step(step.action)
            obs = normalizer(raw_obs)

            # Bootstrap from the observation before the automatic reset.
            final_obs = extras["obs_before_reset"] if extras["obs_before_reset"] is not None else raw_obs
            next_values.append(self.critic_target(normalizer(final_obs)).value)
            rewards.append(reward)
            terminated.append(extras["terminated"].bool())
            truncated.append(extras["truncated"].bool())

            self.track_episodes(reward, done.bool(), finished_returns, finished_lengths)

        self.obs_normalizer.update(torch.cat(raw_obs_buffer))

        rollout = {
            "observations": torch.stack(obs_buffer),
            "rewards": torch.stack(rewards),
            "next_values": torch.stack(next_values),
            "terminated": torch.stack(terminated),
            "truncated": torch.stack(truncated),
        }
        episode_info = {}
        if finished_returns:
            episode_info = {
                "episode/mean_rewards": float(np.mean(finished_returns)),
                "episode/mean_length": float(np.mean(finished_lengths)),
            }
        return rollout, episode_info

    def update_actor(self, rollout:dict) -> dict:
        self.actor_optimizer.zero_grad(set_to_none=True)
        policy_result = SHAC.compute_policy_loss(
            rewards=rollout["rewards"], next_values=rollout["next_values"], terminated=rollout["terminated"],
            gamma=self.gamma, truncated=rollout["truncated"],
        )
        policy_result["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        # Contact-rich rollouts can occasionally produce non-finite gradients; skip those updates.
        if torch.isfinite(grad_norm):
            self.actor_optimizer.step()
        return {
            "update/policy_loss": policy_result["loss"].item(),
            "update/actor_grad_norm": grad_norm.item(),
            "update/mean_reward": policy_result["reward"].item(),
        }

    def update_critic(self, rollout:dict) -> dict:
        targets = SHAC.compute_td_lambda_targets(
            rewards=rollout["rewards"], next_values=rollout["next_values"], terminated=rollout["terminated"],
            gamma=self.gamma, lambda_=self.lambda_, truncated=rollout["truncated"],
        )
        observations = rollout["observations"].reshape(-1, rollout["observations"].shape[-1])
        targets = targets.reshape(-1)
        batch_size = max(1, observations.shape[0] // self.critic_batches)

        losses = []
        for _ in range(self.critic_iterations):
            permutation = torch.randperm(observations.shape[0], device=self.device)
            for start in range(0, observations.shape[0], batch_size):
                index = permutation[start:start + batch_size]
                self.critic_optimizer.zero_grad(set_to_none=True)
                value_result = SHAC.compute_value_loss(
                    self.critic, observations=observations[index], targets=targets[index],
                )
                value_result["loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                losses.append(value_result["loss"].item())

        SHAC.update_target_param(self.critic, self.critic_target, self.tau)
        return {
            "update/value_loss": float(np.mean(losses)),
            "update/value_target": targets.mean().item(),
        }

    def train(self, num_epoch:int, log_interval:int=10):
        for epoch in trange(num_epoch):
            self.set_learning_rates(epoch)
            rollout, episode_info = self.rollout()
            train_info = self.update_actor(rollout)
            train_info.update(self.update_critic(rollout))
            if episode_info:
                self.logger.log_metrics(episode_info, self.global_step)
            if epoch % log_interval == 0 or epoch == num_epoch - 1:
                self.logger.log_metrics(train_info, self.global_step)

    def close(self):
        self.logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SHAC demo on a Rewarped DFlex task.")
    parser.add_argument("--env", default="Ant", choices=sorted(REWARPED_ENVS))
    parser.add_argument("--env-num", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--device", default=None, help="For example: cuda, cuda:0.")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    parser.add_argument("--save-path", default=None,
                        help="Checkpoint path. Defaults to checkpoints/SHAC-<env>-<seed>.pt.")
    args = parser.parse_args()
    for key in ("env_num", "epochs", "horizon"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive.")

    trainer = Trainer(args.env, args.env_num, args.seed, horizon=args.horizon,
                      max_epochs=args.epochs, device=args.device, use_wandb=args.wandb)
    try:
        trainer.train(args.epochs)
        save_checkpoint(
            trainer, args.save_path or default_checkpoint_path("SHAC", args.env, args.seed),
            algorithm="SHAC", env_name=args.env, seed=args.seed,
        )
    finally:
        trainer.close()
