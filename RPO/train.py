import argparse
import copy
import sys
from pathlib import Path

from tqdm import trange

import torch
import torch.optim as optim

import numpy as np

from RLAlg.alg.rpo import RPO
from RLAlg.normalizer import Normalizer
from RLAlg.utils import set_seed_everywhere

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from RPO.model import Actor, EnsembleCritic

from demo_utils import DemoLogger, default_checkpoint_path, save_checkpoint
from SHAC.train import REWARPED_ENVS, make_rewarped_env


class Trainer:
    """RPO on a differentiable simulator with the Rewarped/DFlex env API.

    Same rollout API as the SHAC demo. Each epoch backpropagates once through a
    SAPO-style rollout (soft rewards with a fixed entropy coefficient, critic-mean
    bootstrap), caches the gradient for every action, and reuses it for several
    clipped, KL-regularized policy epochs before training the two critics.
    """

    def __init__(self, env_name:str="Ant", env_num:int=128, seed:int=0, *,
                 horizon:int=32, max_epochs:int=2500, device:str | None=None,
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

        # Network sizes and optimizers follow the paper's Table 2 (shared by all its methods).
        self.actor = Actor(obs_dim, action_dim, [400, 200, 100]).to(self.device)
        self.critic = EnsembleCritic(obs_dim, [400, 200, 100], num_critics=2).to(self.device)

        self.actor_lr = 5e-4
        self.critic_lr = 5e-4
        self.min_lr = 1e-5
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=self.actor_lr, betas=(0.7, 0.95))
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=self.critic_lr, betas=(0.7, 0.95))

        # RPO's Ant coefficients (Table 4); the entropy coefficient is a fixed SAPO temperature.
        self.alpha = 0.2
        self.kl_coef = 0.25
        self.clip_low = 0.8
        self.clip_high = 1.0
        self.target_entropy = -0.5 * action_dim

        self.obs_normalizer = Normalizer(obs_dim).to(self.device)

        self.gamma = 0.99
        self.lambda_ = 0.95
        self.actor_iterations = 5
        self.critic_iterations = 16
        self.critic_batches = 4
        self.max_grad_norm = 0.5

        self.global_step = 0
        self.episode_return = torch.zeros(self.env_num, device=self.device)
        self.episode_length = torch.zeros(self.env_num, device=self.device)
        self.logger = DemoLogger(f"RPO-{env_name}-{seed}", use_wandb)

    def set_learning_rates(self, epoch:int):
        # Exponential decay to 1e-5 over training; the paper names the schedule but not its rate.
        progress = epoch / self.max_epochs
        for optimizer, lr in ((self.actor_optimizer, self.actor_lr), (self.critic_optimizer, self.critic_lr)):
            for group in optimizer.param_groups:
                group["lr"] = lr * (self.min_lr / lr) ** progress

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

        raw_obs_buffer, obs_buffer, actions, rewards, entropies, terminated, truncated = [], [], [], [], [], [], []
        mean_values = []
        finished_returns, finished_lengths = [], []
        for _ in range(self.horizon):
            self.global_step += self.env_num
            raw_obs_buffer.append(raw_obs.detach())
            obs_buffer.append(obs.detach())

            step = self.actor(obs)
            # The action keeps its graph: the cached gradients are taken with respect to it.
            actions.append(step.action)
            raw_obs, reward, done, extras = self.env.step(step.action)
            obs = normalizer(raw_obs)

            # Bootstrap from the observation before the automatic reset with the live critics.
            final_obs = extras["obs_before_reset"] if extras["obs_before_reset"] is not None else raw_obs
            mean_values.append(RPO.compute_next_values(self.critic, normalizer(final_obs))[1])
            rewards.append(reward)
            entropies.append(step.entropy)
            terminated.append(extras["terminated"].bool())
            truncated.append(extras["truncated"].bool())

            self.track_episodes(reward, done.bool(), finished_returns, finished_lengths)

        self.obs_normalizer.update(torch.cat(raw_obs_buffer))

        rollout = {
            "observations": torch.stack(obs_buffer),
            "actions": actions,
            "rewards": torch.stack(rewards),
            "entropies": torch.stack(entropies),
            "next_values": torch.stack(mean_values),
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
        # One BPTT pass through the rollout; this frees its graph.
        cached = RPO.compute_action_gradients(
            actions=rollout["actions"], rewards=rollout["rewards"], entropies=rollout["entropies"],
            next_values=rollout["next_values"], terminated=rollout["terminated"], gamma=self.gamma,
            alpha=self.alpha, target_entropy=self.target_entropy, truncated=rollout["truncated"],
        )
        observations = rollout["observations"]
        actions = torch.stack(rollout["actions"]).detach()
        action_gradients = cached["action_gradients"]
        info = {
            "update/policy_loss": cached["loss"].item(),
            "update/mean_reward": cached["reward"].item(),
            "update/entropy": cached["entropy"].item(),
        }
        # Contact-rich rollouts can occasionally produce non-finite gradients; skip those updates.
        if not torch.isfinite(action_gradients).all():
            info["update/skipped"] = 1.0
            return info

        # Score the stored actions once more so the first epoch's ratio is exactly 1.
        with torch.no_grad():
            behavior = self.actor(observations, actions)
        log_probs_hat = behavior.log_prob
        means_hat, stds_hat = behavior.pi.base_dist.loc, behavior.pi.base_dist.scale

        grad_norms, kls, clip_fractions = [], [], []
        for _ in range(self.actor_iterations):
            policy_result = RPO.compute_policy_loss(
                self.actor, observations=observations, actions=actions, action_gradients=action_gradients,
                log_probs_hat=log_probs_hat, means_hat=means_hat, stds_hat=stds_hat,
                clip_low=self.clip_low, clip_high=self.clip_high, kl_coef=self.kl_coef,
            )
            self.actor_optimizer.zero_grad(set_to_none=True)
            policy_result["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            if torch.isfinite(grad_norm):
                self.actor_optimizer.step()
            grad_norms.append(grad_norm.item())
            kls.append(policy_result["kl_divergence"].item())
            clip_fractions.append(policy_result["policy_clip_fraction"].item())

        info.update({
            "update/actor_grad_norm": float(np.mean(grad_norms)),
            "update/kl_divergence": kls[-1],
            "update/policy_clip_fraction": float(np.mean(clip_fractions)),
            "update/skipped": 0.0,
        })
        return info

    def update_critic(self, rollout:dict) -> dict:
        # The paper computes the targets from the critics' mean (SAPO uses their minimum).
        targets = RPO.compute_td_lambda_targets(
            rewards=rollout["rewards"], entropies=rollout["entropies"], next_values=rollout["next_values"],
            terminated=rollout["terminated"], gamma=self.gamma, lambda_=self.lambda_, alpha=self.alpha,
            target_entropy=self.target_entropy, truncated=rollout["truncated"],
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
                value_result = RPO.compute_value_loss(
                    self.critic, observations=observations[index], targets=targets[index],
                )
                value_result["loss"].backward()
                # As in the SAPO reference, zero non-finite critic gradients from simulation blow-ups.
                for param in self.critic.parameters():
                    param.grad.nan_to_num_(0.0, 0.0, 0.0)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                losses.append(value_result["loss"].item())

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
    parser = argparse.ArgumentParser(description="Train the RPO demo on a Rewarped DFlex task.")
    parser.add_argument("--env", default="Ant", choices=sorted(REWARPED_ENVS))
    parser.add_argument("--env-num", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--device", default=None, help="For example: cuda, cuda:0.")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    parser.add_argument("--save-path", default=None,
                        help="Checkpoint path. Defaults to checkpoints/RPO-<env>-<seed>.pt.")
    args = parser.parse_args()
    for key in ("env_num", "epochs", "horizon"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive.")

    trainer = Trainer(args.env, args.env_num, args.seed, horizon=args.horizon, max_epochs=args.epochs,
                      device=args.device, use_wandb=args.wandb)
    try:
        trainer.train(args.epochs)
        save_checkpoint(
            trainer, args.save_path or default_checkpoint_path("RPO", args.env, args.seed),
            algorithm="RPO", env_name=args.env, seed=args.seed,
        )
    finally:
        trainer.close()
