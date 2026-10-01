"""Collect a small random-policy dataset to exercise the offline IQL demo."""

import argparse
import sys
from pathlib import Path

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_utils import continuous_action_limit, setup_env


def collect_dataset(env_name, steps, seed=0):
    if steps <= 0:
        raise ValueError("steps must be positive.")
    env = setup_env(env_name)
    try:
        continuous_action_limit(env.action_space, "cpu")
        env.action_space.seed(seed)
        obs, _ = env.reset(seed=seed)
        records = {key: [] for key in (
            "observations", "actions", "rewards", "next_observations", "terminated", "truncated",
        )}
        for _ in range(steps):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            for key, value in (
                ("observations", obs.copy()), ("actions", action.copy()),
                ("rewards", reward), ("next_observations", next_obs.copy()),
                ("terminated", terminated), ("truncated", truncated),
            ):
                records[key].append(value)
            if terminated or truncated:
                obs, _ = env.reset()
            else:
                obs = next_obs
        dataset = {key: np.asarray(values) for key, values in records.items()}
        dataset["env_name"] = np.asarray(env_name)
        return dataset
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="Pendulum-v1")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/pendulum_random.npz"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        parser.error("--output must end with .npz")
    if args.output.exists() and not args.overwrite:
        parser.error("Dataset already exists; choose another path or pass --overwrite.")
    if args.steps <= 0:
        parser.error("--steps must be positive.")
    dataset = collect_dataset(args.env, args.steps, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **dataset)
    print(f"Saved {args.steps} transitions to {args.output}")


if __name__ == "__main__":
    main()
