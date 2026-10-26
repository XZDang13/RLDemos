import importlib

import gymnasium as gym
import numpy as np
import pytest
import torch

from RLAlg.nn.steps import ValueStep

from demo_utils import episode_metrics, reset_finished_envs
from IQL.collect import collect_dataset
from IQL.train import Trainer as IQLTrainer


OFF_POLICY = ["DDPG", "DDPG_DQ", "SAC", "DSAC", "DSACT"]


class BoundaryEnv(gym.Env):
    """A terminal observation of [2] and a distinguishable reset observation of [0]."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(-2.0, 2.0, shape=(1,), dtype=np.float32)

    def __init__(self, terminal=False, length=2, render_mode=None):
        self.terminal = terminal
        self.length = length
        self.render_mode = render_mode

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.array([0], dtype=np.float32), {}

    def step(self, action):
        assert self.action_space.contains(action)
        self.steps += 1
        obs = np.array([self.steps], dtype=np.float32)
        finished = self.steps == self.length
        return obs, 1.0, finished and self.terminal, finished and not self.terminal, {}


for terminal in (False, True):
    gym.register(
        id=f"RLDemos-{'Terminal' if terminal else 'Timeout'}-v0",
        entry_point=BoundaryEnv, kwargs={"terminal": terminal}, max_episode_steps=2,
    )


@pytest.fixture(scope="session", autouse=True)
def limit_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def trainer_for(name, env="Pendulum-v1", **options):
    cls = importlib.import_module(f"{name}.train").Trainer
    kwargs = {"rollout_steps": 4, "device": "cpu"}
    if name in OFF_POLICY:
        kwargs["buffer_steps"] = 8
    if name == "FPO":
        kwargs.update(flow_steps=2, cmf_samples=3)
    kwargs.update(options)
    return cls(env, 2, **kwargs)


@pytest.mark.parametrize("name,env,options", [
    ("PPO", "CartPole-v1", {}),
    ("PPO", "Pendulum-v1", {}),
    ("PPO", "Pendulum-v1", {"std_parameterization": "std", "init_std": 0.6}),
    ("FPO", "Pendulum-v1", {}),
    *[(name, "Pendulum-v1", {}) for name in OFF_POLICY],
])
def test_online_rollout_and_update_changes_parameters(name, env, options):
    trainer = trainer_for(name, env, **options)
    try:
        before = {key: value.clone() for key, value in trainer.actor.state_dict().items()}
        trainer.obs, _ = trainer.envs.reset(seed=trainer.seed)
        trainer.rollout()
        metrics = trainer.update(2, 4)
        assert metrics and all(np.isfinite(value) for value in metrics.values())
        assert any(not torch.equal(before[key], value) for key, value in trainer.actor.state_dict().items())
        assert trainer.global_step == 8
        # Also exercise an episode continuing across two rollout windows.
        trainer.rollout()
        assert trainer.global_step == 16
        assert trainer.replay_buffer.current_size == (8 if name in OFF_POLICY else 4)
        if name == "PPO" and env == "CartPole-v1":
            assert trainer.replay_buffer.data["actions"].dtype == torch.long
        for target_name in ("actor_target", "critic_target"):
            if hasattr(trainer, target_name):
                assert all(not p.requires_grad and p.grad is None for p in getattr(trainer, target_name).parameters())
    finally:
        trainer.close()


@pytest.mark.parametrize("name", OFF_POLICY)
@pytest.mark.parametrize("terminal", [False, True])
def test_replay_keeps_final_observation_and_only_masks_true_terminals(name, terminal):
    env = f"RLDemos-{'Terminal' if terminal else 'Timeout'}-v0"
    trainer = trainer_for(name, env)
    try:
        trainer.obs, _ = trainer.envs.reset(seed=0)
        trainer.rollout()
        data = trainer.replay_buffer.data
        assert data["observations"][:4, 0, 0].tolist() == [0, 1, 0, 1]
        assert data["next_observations"][:4, 0, 0].tolist() == [1, 2, 1, 2]
        assert data["dones"][:4, 0].tolist() == [False, terminal, False, terminal]
        assert data["rewards"][:4, 0].tolist() == [1, 1, 1, 1]
    finally:
        trainer.close()


class ObservationValue(torch.nn.Module):
    def forward(self, obs):
        return ValueStep(value=obs[..., 0])


@pytest.mark.parametrize("name", ["PPO", "FPO"])
@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("bootstrap", [False, True])
def test_gae_bootstraps_final_state_without_crossing_episode_boundary(name, terminal, bootstrap):
    env = f"RLDemos-{'Terminal' if terminal else 'Timeout'}-v0"
    trainer = trainer_for(name, env, bootstrap_timeouts=bootstrap)
    try:
        trainer.critic = ObservationValue()
        trainer.obs, _ = trainer.envs.reset(seed=0)
        trainer.rollout()
        data = trainer.replay_buffer.data
        reward = getattr(trainer, "reward_scale", 1.0)
        expected = reward + trainer.gamma * 2 if bootstrap and not terminal else reward
        assert data["returns"][1, 0].item() == pytest.approx(expected)
        assert data["returns"][3, 0].item() == pytest.approx(expected)
        assert data["returns"][0, 0].item() == pytest.approx(data["returns"][2, 0].item())
        assert data["truncated"][:, 0].tolist() == [False, True, False, True]
        assert torch.isfinite(data["advantages"]).all()
    finally:
        trainer.close()


def test_partial_reset_preserves_running_env_and_episode_statistics():
    envs = gym.vector.SyncVectorEnv(
        [lambda length=length: gym.wrappers.RecordEpisodeStatistics(BoundaryEnv(length=length)) for length in (2, 3)],
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
    )
    try:
        envs.reset(seed=0)
        action = np.zeros((2, 1), dtype=np.float32)
        envs.step(action)
        obs, _, terminated, truncated, info = envs.step(action)
        assert episode_metrics(info) == {"episode/mean_rewards": 2.0, "episode/mean_length": 2.0}
        obs = reset_finished_envs(envs, obs, terminated, truncated)
        assert obs[:, 0].tolist() == [0, 2]
        obs, _, terminated, truncated, info = envs.step(action)
        assert episode_metrics(info) == {"episode/mean_rewards": 3.0, "episode/mean_length": 3.0}
        obs = reset_finished_envs(envs, obs, terminated, truncated)
        assert obs[:, 0].tolist() == [1, 0]
    finally:
        envs.close()


def test_double_q_passes_std_and_gamma_independently(monkeypatch):
    from RLAlg.alg.ddpg_double_q import DDPGDoubleQ
    original = DDPGDoubleQ.compute_critic_loss
    seen = {}

    def capture(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(DDPGDoubleQ, "compute_critic_loss", capture)
    trainer = trainer_for("DDPG_DQ")
    try:
        trainer.std = 0.23
        trainer.gamma = 0.8
        trainer.obs, _ = trainer.envs.reset(seed=0)
        trainer.rollout()
        trainer.update(1, 4)
        assert seen["std"] == 0.23 and seen["gamma"] == 0.8
        assert {"observations", "actions", "rewards", "next_observations", "terminated"} <= seen.keys()
    finally:
        trainer.close()


def test_fpo_reuses_stored_cmf_samples():
    from RLAlg.alg.fpo import FPO
    trainer = trainer_for("FPO")
    try:
        trainer.obs, _ = trainer.envs.reset(seed=0)
        trainer.rollout()
        batch = trainer.replay_buffer.sample_batch(trainer.batch_keys, 4)
        losses = FPO.compute_cmf_loss(
            trainer.actor, observations=batch["observations"], actions=batch["actions"],
            eps=batch["eps"], t=batch["time_steps"],
        )
        assert torch.allclose(losses, batch["init_cmf_loss"], atol=1e-6)
    finally:
        trainer.close()


def test_iql_offline_update_does_not_step_env_or_mutate_dataset(tmp_path, monkeypatch):
    dataset = collect_dataset("RLDemos-Timeout-v0", 8)
    assert dataset["next_observations"][:, 0].tolist() == [1, 2] * 4
    assert dataset["observations"][:, 0].tolist() == [0, 1] * 4
    path = tmp_path / "dataset.npz"
    np.savez_compressed(path, **dataset)
    trainer = IQLTrainer("RLDemos-Timeout-v0", path, device="cpu")
    try:
        before_data = {key: value.clone() for key, value in trainer.dataset.items()}
        before_models = [{key: value.clone() for key, value in model.state_dict().items()} for model in (trainer.actor, trainer.critic, trainer.value)]

        def fail_step(*args, **kwargs):
            pytest.fail("Offline updates must not step the environment.")

        monkeypatch.setattr(trainer.env, "step", fail_step)
        metrics = trainer.update(2, 4)
        assert all(np.isfinite(value) for value in metrics.values())
        assert trainer.global_step == 2
        assert all(torch.equal(before_data[key], value) for key, value in trainer.dataset.items())
        for before, model in zip(before_models, (trainer.actor, trainer.critic, trainer.value)):
            assert any(not torch.equal(before[key], value) for key, value in model.state_dict().items())
        assert trainer.dataset["terminated"].sum().item() == 0
        assert trainer.dataset["truncated"].sum().item() == 4
    finally:
        trainer.close()


@pytest.mark.parametrize("problem", ["missing", "shape", "nonfinite", "mask", "bounds", "env"])
def test_iql_rejects_invalid_datasets(problem):
    dataset = collect_dataset("Pendulum-v1", 4)
    trainer = IQLTrainer("Pendulum-v1", dataset, device="cpu")
    try:
        if problem == "missing":
            del dataset["terminated"]
        elif problem == "shape":
            dataset["rewards"] = dataset["rewards"][:, None]
        elif problem == "nonfinite":
            dataset["observations"][0, 0] = np.nan
        elif problem == "mask":
            dataset["terminated"] = np.full(4, 2)
        elif problem == "bounds":
            dataset["actions"][0, 0] = 10
        else:
            dataset["env_name"] = np.asarray("CartPole-v1")
        with pytest.raises(ValueError):
            trainer.load_dataset(dataset)
    finally:
        trainer.close()
