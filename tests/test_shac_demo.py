import pytest
import torch

from SHAC.train import Trainer


class PointMassEnv:
    """Differentiable torch env with the Rewarped API: s' = s + 0.1 a, r = -|s'|^2."""

    def __init__(self, num_envs=4, obs_dim=3, episode_length=5, terminate_below=None):
        self.num_envs = num_envs
        self.num_obs = obs_dim
        self.num_act = obs_dim
        self.episode_length = episode_length
        self.terminate_below = terminate_below
        self.generator = torch.Generator().manual_seed(0)
        self.state = self._initial_state(num_envs)
        self.progress = torch.zeros(num_envs, dtype=torch.long)

    def _initial_state(self, count):
        return torch.rand(count, self.num_obs, generator=self.generator) * 2.0 + 1.0

    def initialize_trajectory(self):
        self.state = self.state.detach()
        return self.state

    def step(self, actions):
        self.state = self.state + 0.1 * actions
        self.progress += 1
        reward = -(self.state ** 2).sum(-1)
        truncated = self.progress >= self.episode_length
        terminated = torch.zeros_like(truncated)
        if self.terminate_below is not None:
            terminated = self.state[:, 0] < self.terminate_below
        done = terminated | truncated
        extras = {"terminated": terminated, "truncated": truncated, "obs_before_reset": None}
        if done.any():
            extras["obs_before_reset"] = self.state.clone()
            state = self.state.clone()
            state[done] = self._initial_state(int(done.sum()))
            self.state = state
            self.progress[done] = 0
        return self.state, reward, done, extras


def make_trainer(env, **options):
    return Trainer("PointMass", env.num_envs, seed=0, horizon=4, max_epochs=10, device="cpu", env=env, **options)


def test_rollout_shapes_and_boundaries():
    env = PointMassEnv(episode_length=3)
    trainer = make_trainer(env)

    rollout, episode_info = trainer.rollout()

    for key in ("observations",):
        assert rollout[key].shape == (4, env.num_envs, env.num_obs)
    for key in ("rewards", "next_values", "terminated", "truncated"):
        assert rollout[key].shape == (4, env.num_envs)
    assert rollout["rewards"].requires_grad and rollout["next_values"].requires_grad
    assert not rollout["observations"].requires_grad
    assert rollout["truncated"][2].all() and not rollout["truncated"][[0, 1, 3]].any()
    assert episode_info["episode/mean_length"] == 3.0
    assert trainer.obs_normalizer.count.item() == 4 * env.num_envs


def test_bootstrap_uses_observation_before_reset():
    # Every step ends an episode, so every bootstrap must use the pre-reset state.
    env = PointMassEnv(num_envs=2, episode_length=1)
    trainer = make_trainer(env)
    critic_inputs, final_obs, reset_obs = [], [], []
    critic_forward, env_step = trainer.critic_target.forward, env.step

    def record_critic(x):
        critic_inputs.append(x.detach().clone())
        return critic_forward(x)

    def record_step(actions):
        obs, reward, done, extras = env_step(actions)
        final_obs.append(extras["obs_before_reset"].detach().clone())
        reset_obs.append(obs.detach().clone())
        return obs, reward, done, extras

    trainer.critic_target.forward = record_critic
    env.step = record_step
    trainer.rollout()

    # Statistics start at mean 0 and variance 1, so the frozen normalization is the identity.
    assert len(critic_inputs) == len(final_obs) == 4
    for seen, final, reset in zip(critic_inputs, final_obs, reset_obs):
        torch.testing.assert_close(seen, final)
        assert not torch.allclose(seen, reset)


def test_actor_and_critic_updates_change_parameters():
    env = PointMassEnv(terminate_below=1.2)
    trainer = make_trainer(env)
    actor_before = [p.detach().clone() for p in trainer.actor.parameters()]
    critic_before = [p.detach().clone() for p in trainer.critic.parameters()]
    target_before = [p.detach().clone() for p in trainer.critic_target.parameters()]

    rollout, _ = trainer.rollout()
    actor_info = trainer.update_actor(rollout)
    critic_info = trainer.update_critic(rollout)

    assert all(torch.isfinite(torch.tensor(v)) for v in {**actor_info, **critic_info}.values())
    assert any(not torch.equal(a, p) for a, p in zip(actor_before, trainer.actor.parameters()))
    assert any(not torch.equal(a, p) for a, p in zip(critic_before, trainer.critic.parameters()))
    assert any(not torch.equal(a, p) for a, p in zip(target_before, trainer.critic_target.parameters()))
    assert all(p.grad is None or torch.all(p.grad == 0) for p in trainer.critic_target.parameters())


def test_learning_rate_decays_linearly():
    trainer = make_trainer(PointMassEnv())

    trainer.set_learning_rates(5)

    assert trainer.actor_optimizer.param_groups[0]["lr"] == pytest.approx((1e-5 + 2e-3) / 2)
    assert trainer.critic_optimizer.param_groups[0]["lr"] == pytest.approx((1e-5 + 5e-4) / 2)


def test_training_improves_point_mass_return():
    torch.manual_seed(0)
    env = PointMassEnv(num_envs=16, episode_length=1000)
    trainer = make_trainer(env)
    trainer.max_epochs = 60

    def mean_reward():
        rollout, _ = trainer.rollout()
        return rollout["rewards"].mean().item()

    with torch.no_grad():
        env.state = env._initial_state(env.num_envs)
    before = mean_reward()
    trainer.train(60, log_interval=1000)
    with torch.no_grad():
        env.state = env._initial_state(env.num_envs)
    after = mean_reward()

    assert after > before


def test_rewarped_ant_smoke():
    pytest.importorskip("rewarped")
    if not torch.cuda.is_available():
        pytest.skip("Rewarped tasks run on CUDA.")
    trainer = Trainer("Ant", 4, seed=0, horizon=4, max_epochs=1, device="cuda")

    rollout, _ = trainer.rollout()
    info = trainer.update_actor(rollout)
    info.update(trainer.update_critic(rollout))

    assert rollout["rewards"].shape == (4, 4)
    assert info["update/actor_grad_norm"] > 0
    assert all(torch.isfinite(torch.tensor(v)) for v in info.values())
