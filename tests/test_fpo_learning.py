"""Behavioral checks for the RLAlg FPO implementation and its runnable demo."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from FPO.train import Trainer
from RLAlg.alg.fpo import FPO, SuperviseTarget, TimeStepSamplerStrategy
from RLAlg.nn.steps import ValueStep


class TargetBandit(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
    metadata = {"render_modes": []}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        assert self.action_space.contains(action)
        reward = -float((action[0] - 0.6) ** 2)
        return np.zeros(1, dtype=np.float32), reward, True, False, {}


gym.register("RLDemos-FPOLearning-v0", entry_point=TargetBandit, max_episode_steps=1)


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@torch.no_grad()
def policy_reward(trainer, noise):
    observations = torch.zeros((len(noise), 1))
    step = FPO.sample_actions(trainer.actor, observations, noise, trainer.flow_steps)
    actions = torch.tanh(step.action)
    return -((actions - 0.6) ** 2).mean().item()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fpo_learns_rewarded_actions_from_actual_rollouts(seed):
    trainer = Trainer(
        "RLDemos-FPOLearning-v0", 8, seed=seed, rollout_steps=32,
        flow_steps=4, cmf_samples=8, device="cpu",
    )
    try:
        noise = torch.randn(2048, 1, generator=torch.Generator().manual_seed(987))
        before = policy_reward(trainer, noise)
        trainer.obs, _ = trainer.envs.reset(seed=seed)
        for _ in range(10):
            trainer.rollout()
            metrics = trainer.update(4, 128)
            assert all(np.isfinite(value) for value in metrics.values())
        after = policy_reward(trainer, noise)
        assert after > before + 0.3
        assert after > -0.25
    finally:
        trainer.close()


@pytest.mark.parametrize("target", ["noise", "velocity"])
@pytest.mark.parametrize("sampler", ["continuous", "discrete"])
def test_rollout_and_update_use_the_trainer_configuration(target, sampler, monkeypatch):
    trainer = Trainer(
        "Pendulum-v1", 2, rollout_steps=4, flow_steps=4, cmf_samples=3,
        supervise_target=target, time_step_sampler=sampler, device="cpu",
    )
    try:
        # Another caller changing legacy defaults must not change this run.
        monkeypatch.setattr(FPO, "supervise_target", SuperviseTarget.Velocity if target == "noise" else SuperviseTarget.Noise)
        monkeypatch.setattr(FPO, "time_step_sampler", TimeStepSamplerStrategy.Discrete if sampler == "continuous" else TimeStepSamplerStrategy.Continuous)
        trainer.obs, _ = trainer.envs.reset(seed=0)
        trainer.rollout()
        batch = trainer.replay_buffer.sample_batch(trainer.batch_keys, 8)
        losses = FPO.compute_cmf_loss(
            trainer.actor, batch["observations"], batch["actions"], batch["eps"], batch["time_steps"],
            supervise_target=trainer.supervise_target,
        )
        assert torch.allclose(losses, batch["init_cmf_loss"], atol=1e-6)
        result = FPO.compute_policy_loss(
            trainer.actor, batch["observations"], batch["actions"], batch["eps"], batch["time_steps"],
            batch["init_cmf_loss"], batch["advantages"], trainer.clip_ratio,
            supervise_target=trainer.supervise_target,
        )
        assert result["ratio"].item() == pytest.approx(1.0)
        assert result["clip_fraction"].item() == pytest.approx(0.0)
        assert all(np.isfinite(value) for value in trainer.update(1, 8).values())
    finally:
        trainer.close()


def test_rejects_discrete_noise_loss_that_has_no_learning_signal():
    with pytest.raises(ValueError, match="zero noise-loss gradients"):
        Trainer("Pendulum-v1", 2, flow_steps=1, time_step_sampler="discrete")


def test_normalization_is_frozen_until_the_next_rollout():
    trainer = Trainer("Pendulum-v1", 2, rollout_steps=8, cmf_samples=3, device="cpu")
    try:
        trainer.obs, _ = trainer.envs.reset(seed=0)
        trainer.rollout()
        assert trainer.observation_normalizer.count.item() == 0
        previous_observations = trainer.replay_buffer.data["observations"].clone()
        trainer.update(1, 16)
        assert trainer.observation_normalizer.count.item() == 0
        trainer.rollout()
        assert trainer.observation_normalizer.count.item() == 16
        assert torch.allclose(trainer.observation_normalizer.mean.float(), previous_observations.mean((0, 1)))
        batch = trainer.replay_buffer.sample_batch(trainer.batch_keys, 16)
        current = FPO.compute_cmf_loss(
            trainer.actor, batch["observations"], batch["actions"], batch["eps"], batch["time_steps"],
            supervise_target=trainer.supervise_target,
        )
        assert torch.allclose(current, batch["init_cmf_loss"], atol=1e-6)
        before = trainer.observation_normalizer.mean.clone()
        trainer.update(2, 8)
        assert torch.equal(trainer.observation_normalizer.mean, before)
        assert trainer.observation_normalizer.count.item() == 16
        optimized = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
        assert all(id(p) in optimized for p in trainer.actor.parameters())
        assert all(id(p) in optimized for p in trainer.critic.parameters())
    finally:
        trainer.close()


class ScalarValue(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(float(value)))

    def forward(self, obs):
        return ValueStep(self.value.expand(obs.shape[:-1]))


def test_value_learning_continues_beyond_the_policy_clip_range():
    trainer = Trainer("RLDemos-FPOLearning-v0", 2, rollout_steps=2, device="cpu")
    try:
        trainer.obs, _ = trainer.envs.reset(seed=0)
        trainer.rollout()
        trainer.critic = ScalarValue(1.0)
        trainer.optimizer = torch.optim.SGD(
            list(trainer.actor.parameters()) + list(trainer.critic.parameters()), lr=0.1,
        )
        # The critic has already moved further than the policy's 0.05 clip,
        # but is still far from the return. Reusing that clip gives no gradient.
        trainer.replay_buffer.data["values"].zero_()
        trainer.replay_buffer.data["returns"].fill_(10.0)
        trainer.replay_buffer.data["advantages"].zero_()
        trainer.update(1, 4)
        assert trainer.critic.value.item() > 1.0
    finally:
        trainer.close()


def test_timeout_value_is_not_scaled_a_second_time():
    trainer = Trainer("Pendulum-v1", 2, rollout_steps=2, reward_scale=0.1, device="cpu")
    try:
        trainer.critic = ScalarValue(5.0)
        rewards = np.array([-10.0, -20.0])
        corrected = trainer.timeout_rewards(
            rewards, np.zeros((2, 3), dtype=np.float32),
            np.array([False, True]), np.array([True, False]),
        )
        np.testing.assert_allclose(corrected, [-1.0 + trainer.gamma * 5.0, -2.0])
        np.testing.assert_array_equal(rewards, [-10.0, -20.0])
    finally:
        trainer.close()


def test_separate_evaluation_preserves_training_randomness_and_statistics():
    from FPO.benchmark import evaluate

    trainer = Trainer("RLDemos-FPOLearning-v0", 2, rollout_steps=2, device="cpu")
    try:
        random_state = torch.get_rng_state().clone()
        before = {name: value.clone() for name, value in trainer.actor.state_dict().items()}
        returns = evaluate(trainer, episodes=4)
        assert returns == evaluate(trainer, episodes=4)
        assert torch.equal(torch.get_rng_state(), random_state)
        assert all(torch.equal(value, before[name]) for name, value in trainer.actor.state_dict().items())
        assert trainer.global_step == 0
    finally:
        trainer.close()
