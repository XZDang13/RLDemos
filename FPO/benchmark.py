"""Measure FPO learning with separate, reproducible evaluation episodes."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from FPO.train import Trainer
from RLAlg.alg.fpo import FPO
from demo_utils import make_vector_env, reset_finished_envs


class BenchmarkLogger:
    def __init__(self):
        self.returns = []

    def log_metrics(self, metrics, step):
        if "episode/mean_rewards" in metrics:
            self.returns.append(metrics["episode/mean_rewards"])

    def finish(self):
        pass


@torch.no_grad()
def evaluate(trainer, episodes=4):
    """Use the same evaluation resets/noise at every checkpoint, outside training."""
    envs = make_vector_env(trainer.env_name, episodes)
    rng = torch.Generator(device=trainer.device).manual_seed(919)
    obs, _ = envs.reset(seed=801)
    totals = np.zeros(episodes)
    active = np.ones(episodes, dtype=bool)
    try:
        while active.any():
            observations = torch.as_tensor(obs, dtype=torch.float32, device=trainer.device)
            noise = torch.randn(episodes, trainer.max_action.numel(), generator=rng, device=trainer.device)
            step = FPO.sample_actions(trainer.actor, observations, noise, trainer.flow_steps, deterministic=True)
            actions = (torch.tanh(step.action) * trainer.max_action).cpu().numpy()
            obs, reward, terminated, truncated, _ = envs.step(actions)
            totals += reward * active
            active &= ~(terminated | truncated)
            obs = reset_finished_envs(envs, obs, terminated, truncated)
    finally:
        envs.close()
    return totals.tolist()


def run(args):
    torch.set_num_threads(args.threads)
    trainer = Trainer(
        args.env, args.env_num, seed=args.seed, device=args.device,
        rollout_steps=args.rollout_steps, flow_steps=args.flow_steps,
        cmf_samples=args.cmf_samples, clip_ratio=args.clip_ratio,
        normalize_observations=args.normalize_observations,
        reward_scale=args.reward_scale, value_clip_range=args.value_clip_range,
    )
    trainer.logger = BenchmarkLogger()
    trainer.obs, _ = trainer.envs.reset(seed=args.seed)
    snapshots = []
    started = time.monotonic()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint(epoch, metrics=None, change=None):
        returns = evaluate(trainer, args.eval_episodes)
        record = {
            "epoch": epoch, "steps": trainer.global_step,
            "eval_returns": returns, "eval_mean": float(np.mean(returns)),
            "seconds": time.monotonic() - started,
            "train_mean": float(np.mean(trainer.logger.returns[-20:])) if trainer.logger.returns else None,
            "parameter_change": change, "metrics": metrics,
        }
        snapshots.append(record)
        output.write_text(json.dumps({"config": vars(args), "snapshots": snapshots}, indent=2) + "\n")
        print(f"epoch={epoch} steps={trainer.global_step} eval_return={record['eval_mean']:.2f}", flush=True)

    try:
        checkpoint(0)
        for epoch in range(1, args.epochs + 1):
            trainer.rollout()
            before = torch.cat([p.detach().flatten() for p in trainer.actor.parameters()]).clone()
            metrics = trainer.update(args.iterations, args.batch_size)
            if not all(np.isfinite(value) for value in metrics.values()):
                raise RuntimeError(f"Non-finite training metrics at epoch {epoch}: {metrics}")
            after = torch.cat([p.detach().flatten() for p in trainer.actor.parameters()])
            change = float((after - before).norm())
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                checkpoint(epoch, metrics, change)
    finally:
        trainer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="HalfCheetah-v5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--env-num", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--cmf-samples", type=int, default=10)
    parser.add_argument("--clip-ratio", type=float, default=0.05)
    parser.add_argument("--normalize-observations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reward-scale", type=float, default=0.1)
    parser.add_argument("--value-clip-range", type=float, default=None)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for key in ("epochs", "env_num", "rollout_steps", "iterations", "batch_size", "flow_steps", "cmf_samples", "eval_every", "eval_episodes", "threads"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive.")
    run(args)


if __name__ == "__main__":
    main()
