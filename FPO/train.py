import sys
import math
from collections import defaultdict
from pathlib import Path

import torch

from RLAlg.alg.fpo import FPO, SuperviseTarget, TimeStepSamplerStrategy
from RLAlg.normalizer import Normalizer

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FPO.model import Actor, NormalizedModel
from PPO.train import Trainer as PPOTrainer
from demo_utils import (
    continuous_action_limit, episode_metrics, mean_metrics,
    reset_finished_envs, run_training,
)


class Trainer(PPOTrainer):
    algorithm_name = "FPO"

    def __init__(self, env_name, env_num, seed=0, *, rollout_steps=None,
                 device=None, use_wandb=False, bootstrap_timeouts=True,
                 flow_steps=8, cmf_samples=10, clip_ratio=0.05,
                 supervise_target="noise", time_step_sampler="continuous",
                 normalize_observations=True, reward_scale=0.1, value_clip_range=None):
        if flow_steps <= 0 or cmf_samples <= 0:
            raise ValueError("flow_steps and cmf_samples must be positive.")
        if not 0 < clip_ratio < 1:
            raise ValueError("clip_ratio must be in (0, 1).")
        if not math.isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError("reward_scale must be finite and positive.")
        if value_clip_range is not None and (not math.isfinite(value_clip_range) or value_clip_range <= 0):
            raise ValueError("value_clip_range must be finite and positive.")
        targets = {"noise": SuperviseTarget.Noise, "velocity": SuperviseTarget.Velocity}
        samplers = {
            "continuous": TimeStepSamplerStrategy.Continuous,
            "discrete": TimeStepSamplerStrategy.Discrete,
        }
        if supervise_target not in targets or time_step_sampler not in samplers:
            raise ValueError("Invalid FPO supervision target or timestep sampler.")
        if supervise_target == "noise" and time_step_sampler == "discrete" and flow_steps == 1:
            raise ValueError("A single discrete t=1 sample gives zero noise-loss gradients; use continuous timesteps.")
        self.supervise_target = targets[supervise_target]
        self.time_step_sampler = samplers[time_step_sampler]
        self.flow_steps = flow_steps
        self.cmf_samples = cmf_samples
        self.reward_scale = reward_scale
        self.value_clip_range = value_clip_range
        super().__init__(
            env_name, env_num, seed, rollout_steps=rollout_steps, device=device,
            use_wandb=use_wandb, bootstrap_timeouts=bootstrap_timeouts,
        )
        self.clip_ratio = clip_ratio
        self.entropy_weight = 0.0
        self.observation_normalizer = None
        if normalize_observations:
            self.observation_normalizer = Normalizer(self.envs.single_observation_space.shape).to(self.device)
            # Wrapping preserves the parameters already held by the optimizer.
            self.actor = NormalizedModel(self.actor, self.observation_normalizer)
            self.critic = NormalizedModel(self.critic, self.observation_normalizer)
        action_dim = self.max_action.numel()
        del self.replay_buffer.data["log_probs"]
        for key, shape in (
            ("eps", (cmf_samples, action_dim)),
            ("time_steps", (cmf_samples, 1)),
            ("init_cmf_loss", (cmf_samples,)),
        ):
            self.replay_buffer.create_storage_space(key, shape, torch.float32)
        self.batch_keys = [
            "observations", "actions", "values", "returns", "advantages",
            "eps", "time_steps", "init_cmf_loss",
        ]

    def create_actor(self, obs_dim, action_space, std_parameterization, init_std):
        self.max_action = continuous_action_limit(action_space, self.device)
        actor = Actor(obs_dim, self.max_action.numel(), [128, 128]).to(self.device)
        return actor, torch.float32

    @torch.no_grad()
    def get_action(self, obs):
        observations = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        noise = torch.randn((len(obs), self.max_action.numel()), device=self.device)
        step = FPO.sample_actions_with_cmf_info(
            self.actor,
            observations=observations,
            init_noise=noise,
            flow_steps=self.flow_steps,
            deterministic=True,
            n_samples_per_action=self.cmf_samples,
            supervise_target=self.supervise_target,
            time_step_sampler=self.time_step_sampler,
        )
        # Keep latent actions and their CMF targets together in the buffer.
        # Only the environment action is mapped into the Box bounds.
        action = torch.tanh(step.action) * self.max_action
        value = self.critic(observations).value
        return action.cpu().numpy(), step, value

    def rollout(self):
        if self.observation_normalizer is not None and self.replay_buffer.current_size:
            # Change statistics only before collecting a new rollout. Changing
            # them between old/new CMF losses would invalidate their ratio.
            self.observation_normalizer.update(
                self.replay_buffer.data["observations"][:self.replay_buffer.current_size]
            )
        self.replay_buffer.reset()
        obs = self.obs
        for _ in range(self.rollout_steps):
            action, step, value = self.get_action(obs)
            next_obs, reward, terminated, truncated, info = self.envs.step(action)
            self.global_step += self.env_num
            self.replay_buffer.add_records({
                "observations": obs,
                "actions": step.action,
                "eps": step.eps,
                "time_steps": step.time_step,
                "init_cmf_loss": step.init_cmf_loss,
                "rewards": self.timeout_rewards(reward, next_obs, terminated, truncated),
                "values": value,
                "terminated": terminated,
                "truncated": truncated,
            })
            self.logger.log_metrics(episode_metrics(info), self.global_step)
            obs = reset_finished_envs(self.envs, next_obs, terminated, truncated)
        self.obs = obs
        self.finish_rollout(obs)

    def timeout_rewards(self, rewards, next_obs, terminated, truncated):
        # Scale environment rewards before adding V(final_obs), which is
        # already trained in the scaled units. Episode logs use raw rewards.
        return super().timeout_rewards(
            rewards * self.reward_scale, next_obs, terminated, truncated,
        )

    def update(self, num_iteration, batch_size):
        metrics = defaultdict(list)
        for _ in range(num_iteration):
            for batch in self.replay_buffer.sample_batchs(self.batch_keys, batch_size):
                policy = FPO.compute_policy_loss(
                    self.actor,
                    observations=batch["observations"],
                    actions=batch["actions"],
                    eps=batch["eps"],
                    t=batch["time_steps"],
                    init_cmf_loss=batch["init_cmf_loss"],
                    advantages=batch["advantages"],
                    clip_ratio=self.clip_ratio,
                    average_losses_before_exp=True,
                    supervise_target=self.supervise_target,
                )
                if self.value_clip_range is None:
                    value = FPO.compute_value_loss(
                        self.critic, observations=batch["observations"], returns=batch["returns"],
                    )
                else:
                    value = FPO.compute_clipped_value_loss(
                        self.critic,
                        observations=batch["observations"],
                        values_hat=batch["values"],
                        returns=batch["returns"],
                        clip_ratio=self.value_clip_range,
                    )
                loss = policy["loss"] + self.value_loss_weight * value["loss"]
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                metrics["update/avg_value_loss"].append(value["loss"].detach().item())
                for key, metric in policy.items():
                    if key == "loss":
                        continue
                    metrics[f"update/avg_{key}"].append(metric.detach().item())
        train_info = mean_metrics(metrics)
        self.logger.log_metrics(train_info, self.global_step)
        return train_info


if __name__ == "__main__":
    run_training(Trainer, "FPO", default_iterations=10)
