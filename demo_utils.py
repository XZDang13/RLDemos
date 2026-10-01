"""Shared environment, logging, and command-line helpers for the demos."""

import argparse

import gymnasium as gym
import numpy as np
import torch

from RLAlg.logger import WandbLogger


def setup_env(env_name, mode=None):
    env = gym.make(env_name, render_mode=mode)
    env = gym.wrappers.FlattenObservation(env)
    return gym.wrappers.RecordEpisodeStatistics(env)


def make_vector_env(env_name, env_num):
    if env_num <= 0:
        raise ValueError("env_num must be positive.")
    # Record the terminal observation before explicitly resetting finished envs.
    return gym.vector.SyncVectorEnv(
        [lambda: setup_env(env_name) for _ in range(env_num)],
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
    )


def reset_finished_envs(envs, next_obs, terminated, truncated):
    finished = terminated | truncated
    if finished.any():
        next_obs, _ = envs.reset(options={"reset_mask": finished.copy()})
    return next_obs


def continuous_action_limit(space, device):
    if not isinstance(space, gym.spaces.Box) or len(space.shape) != 1:
        raise ValueError("Continuous demos require a one-dimensional Box action space.")
    if (
        not np.isfinite(space.high).all()
        or not np.isfinite(space.low).all()
        or not np.all(space.high > 0)
        or not np.allclose(space.low, -space.high)
    ):
        raise ValueError("Continuous demos require finite, symmetric action bounds.")
    return torch.as_tensor(space.high, dtype=torch.float32, device=device)


def episode_metrics(info):
    if "episode" not in info:
        return {}
    episode = info["episode"]
    finished = info.get("_episode", episode.get("_r"))
    if finished is None:
        finished = np.ones_like(episode["r"], dtype=bool)
    if not np.any(finished):
        return {}
    return {
        "episode/mean_rewards": float(np.mean(episode["r"][finished])),
        "episode/mean_length": float(np.mean(episode["l"][finished])),
    }


def mean_metrics(metrics):
    return {key: float(np.mean(values)) for key, values in metrics.items()}


class DemoLogger:
    def __init__(self, name, use_wandb=False):
        self.use_wandb = use_wandb
        if use_wandb:
            WandbLogger.init_project("RLDemos", name)

    def log_metrics(self, metrics, step):
        if not metrics:
            return
        if self.use_wandb:
            WandbLogger.log_metrics(metrics, step)
        else:
            values = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
            print(f"step={step} {values}")

    def finish(self):
        if self.use_wandb:
            WandbLogger.finish_project()


def run_training(trainer_type, algorithm, default_iterations=250):
    parser = argparse.ArgumentParser(description=f"Train the {algorithm} demo.")
    parser.add_argument("--env", default="HalfCheetah-v5")
    parser.add_argument("--env-num", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=default_iterations)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--device", default=None, help="For example: cpu, cuda, cuda:0.")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    if algorithm in {"PPO", "FPO"}:
        parser.add_argument("--bootstrap-timeouts", action=argparse.BooleanOptionalAction, default=True)
    if algorithm == "PPO":
        parser.add_argument("--std-parameterization", choices=["log_std", "std"], default="log_std")
        parser.add_argument("--init-std", type=float, default=1.0)
    elif algorithm == "FPO":
        parser.add_argument("--flow-steps", type=int, default=8)
        parser.add_argument("--cmf-samples", type=int, default=10)
        parser.add_argument("--clip-ratio", type=float, default=0.05)
        parser.add_argument("--supervise-target", choices=["noise", "velocity"], default="noise")
        parser.add_argument("--time-step-sampler", choices=["continuous", "discrete"], default="continuous")
        parser.add_argument("--normalize-observations", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--reward-scale", type=float, default=0.1)
        parser.add_argument("--value-clip-range", type=float, default=None)
        parser.add_argument("--threads", type=int, default=1)
    else:
        parser.add_argument("--buffer-steps", type=int, default=100000)
    args = parser.parse_args()
    for key in ("env_num", "epochs", "iterations", "batch_size", "rollout_steps", "buffer_steps", "flow_steps", "cmf_samples", "threads"):
        value = getattr(args, key, None)
        if value is not None and value <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive.")

    options = dict(
        seed=args.seed,
        device=args.device,
        use_wandb=args.wandb,
        rollout_steps=args.rollout_steps,
    )
    if algorithm == "PPO":
        options.update(
            std_parameterization=args.std_parameterization,
            init_std=args.init_std,
            bootstrap_timeouts=args.bootstrap_timeouts,
        )
    elif algorithm == "FPO":
        torch.set_num_threads(args.threads)
        options.update(
            flow_steps=args.flow_steps,
            cmf_samples=args.cmf_samples,
            bootstrap_timeouts=args.bootstrap_timeouts,
            clip_ratio=args.clip_ratio,
            supervise_target=args.supervise_target,
            time_step_sampler=args.time_step_sampler,
            normalize_observations=args.normalize_observations,
            reward_scale=args.reward_scale,
            value_clip_range=args.value_clip_range,
        )
    else:
        options["buffer_steps"] = args.buffer_steps
    trainer = trainer_type(args.env, args.env_num, **options)
    try:
        trainer.train(args.epochs, args.iterations, args.batch_size)
    finally:
        trainer.close()
