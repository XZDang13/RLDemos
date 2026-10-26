import importlib
import sys

import numpy as np
import pytest
import torch

from demo_utils import checkpoint_state, default_checkpoint_path, load_checkpoint, save_checkpoint
from IQL.collect import collect_dataset
from IQL.train import Trainer as IQLTrainer
from SHAC.train import Trainer as SHACTrainer
from test_demos import OFF_POLICY, trainer_for
from test_shac_demo import PointMassEnv


def assert_same_state(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor)
        torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_same_state(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_same_state(x, y)
    else:
        assert a == b


def assert_round_trip(trained, fresh, path):
    expected = checkpoint_state(trained)
    assert any(not torch.equal(p, q) for p, q in zip(trained.actor.parameters(), fresh.actor.parameters()))

    metadata = load_checkpoint(fresh, path)

    assert metadata["global_step"] == trained.global_step
    assert_same_state(checkpoint_state(fresh), expected)


@pytest.mark.parametrize("name", ["PPO", "FPO", *OFF_POLICY])
def test_online_checkpoint_round_trip(name, tmp_path):
    trained, fresh = trainer_for(name, seed=0), trainer_for(name, seed=1)
    try:
        trained.obs, _ = trained.envs.reset(seed=trained.seed)
        trained.rollout()
        trained.update(1, 4)
        path = save_checkpoint(trained, tmp_path / "ckpt.pt", algorithm=name, env_name="Pendulum-v1", seed=0)

        state = torch.load(path)["state"]
        # Optimizer moments are saved so training can resume; environments and buffers are not.
        assert "actor" in state and any("optimizer" in key for key in state)
        assert not {"envs", "replay_buffer", "logger"} & state.keys()
        assert torch.load(path)["metadata"]["algorithm"] == name
        assert_round_trip(trained, fresh, path)
    finally:
        trained.close()
        fresh.close()


def test_iql_checkpoint_round_trip(tmp_path):
    dataset_path = tmp_path / "dataset.npz"
    np.savez_compressed(dataset_path, **collect_dataset("RLDemos-Timeout-v0", 8))
    trained = IQLTrainer("RLDemos-Timeout-v0", dataset_path, seed=0, device="cpu")
    fresh = IQLTrainer("RLDemos-Timeout-v0", dataset_path, seed=1, device="cpu")
    try:
        trained.update(1, 4)
        path = save_checkpoint(trained, tmp_path / "ckpt.pt")
        assert {"actor", "critic", "value", "critic_target"} <= torch.load(path)["state"].keys()
        assert_round_trip(trained, fresh, path)
    finally:
        trained.close()
        fresh.close()


def test_shac_checkpoint_round_trip(tmp_path):
    def make(seed):
        return SHACTrainer("PointMass", 4, seed=seed, horizon=4, device="cpu", env=PointMassEnv())

    trained, fresh = make(0), make(1)
    rollout, _ = trained.rollout()
    trained.update_actor(rollout)
    trained.update_critic(rollout)
    path = save_checkpoint(trained, tmp_path / "ckpt.pt")

    assert {"actor", "critic", "critic_target", "obs_normalizer"} <= torch.load(path)["state"].keys()
    assert_round_trip(trained, fresh, path)
    assert trained.obs_normalizer.count.item() > 0


def test_dsact_running_std_statistics_are_checkpointed(tmp_path):
    trained, fresh = trainer_for("DSACT", seed=0), trainer_for("DSACT", seed=1)
    try:
        trained.obs, _ = trained.envs.reset(seed=trained.seed)
        trained.rollout()
        trained.update(2, 4)
        assert trained.q_mean_std is not None and fresh.q_mean_std is None
        path = save_checkpoint(trained, tmp_path / "ckpt.pt")

        load_checkpoint(fresh, path)

        torch.testing.assert_close(fresh.q_mean_std, trained.q_mean_std)
    finally:
        trained.close()
        fresh.close()


def test_free_learnable_tensors_are_saved_and_restored(tmp_path):
    class Holder:
        def __init__(self, value):
            self.log_alpha = torch.tensor(value, requires_grad=True)
            self.stats = torch.tensor(value)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=0.1)

    trained = Holder(1.0)
    trained.log_alpha.grad = torch.tensor(1.0)
    trained.alpha_optimizer.step()
    path = save_checkpoint(trained, tmp_path / "ckpt.pt")
    fresh = Holder(5.0)

    load_checkpoint(fresh, path)

    assert torch.load(path)["state"].keys() == {"log_alpha", "alpha_optimizer"}
    assert fresh.log_alpha.item() == trained.log_alpha.item() and fresh.log_alpha.requires_grad
    assert fresh.stats.item() == 5.0
    assert_same_state(fresh.alpha_optimizer.state_dict(), trained.alpha_optimizer.state_dict())


def test_cli_saves_checkpoint_after_training(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "sac.pt"
    monkeypatch.setattr(sys, "argv", [
        "SAC.train", "--env", "Pendulum-v1", "--env-num", "2", "--epochs", "1", "--iterations", "1",
        "--rollout-steps", "4", "--buffer-steps", "8", "--batch-size", "4", "--device", "cpu",
        "--save-path", str(path),
    ])

    importlib.import_module("demo_utils").run_training(importlib.import_module("SAC.train").Trainer, "SAC")

    checkpoint = torch.load(path)
    assert checkpoint["metadata"] == {"algorithm": "SAC", "env_name": "Pendulum-v1", "seed": 0, "global_step": 8}
    assert default_checkpoint_path("SAC", "Pendulum-v1", 0).name == "SAC-Pendulum-v1-0.pt"
