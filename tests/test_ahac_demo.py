import pytest
import torch

from AHAC.train import Trainer
from test_shac_demo import PointMassEnv


class ScaledActionProbe:
    """Fake contact probe: per-env force proportional to the last state, unit accelerations."""

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, env):
        forces = self.scale * env.state.detach().abs().unsqueeze(1)
        return forces, torch.ones_like(forces)


def make_trainer(env, scale=100.0, **options):
    return Trainer("PointMass", env.num_envs, seed=0, min_horizon=4, max_horizon=12, max_epochs=10,
                   device="cpu", env=env, contact_probe=ScaledActionProbe(scale), **options)


def test_rollout_length_follows_dual_and_records_contact_norms():
    env = PointMassEnv(episode_length=3)
    trainer = make_trainer(env)

    rollout, episode_info = trainer.rollout()

    assert len(trainer.dual) == 4
    for key in ("rewards", "next_values", "terminated", "truncated", "contact_norms"):
        assert rollout[key].shape == (4, env.num_envs)
    assert rollout["rewards"].requires_grad and rollout["next_values"].requires_grad
    assert not rollout["contact_norms"].requires_grad
    assert (rollout["contact_norms"] > 0).all()
    assert episode_info["episode/mean_length"] == 3.0


def test_next_values_use_minimum_of_live_double_critic():
    env = PointMassEnv(num_envs=2)
    trainer = make_trainer(env)
    seen = []
    forward = trainer.critic.forward

    def record(x):
        steps = forward(x)
        seen.append(torch.min(steps[0].value, steps[1].value).detach())
        return steps

    trainer.critic.forward = record
    rollout, _ = trainer.rollout()

    torch.testing.assert_close(rollout["next_values"].detach(), torch.stack(seen))


def test_update_changes_parameters_and_horizon_tracks_contact():
    stiff = make_trainer(PointMassEnv(terminate_below=1.2), scale=1e4)
    soft = make_trainer(PointMassEnv(terminate_below=1.2), scale=1.0)
    actor_before = [p.detach().clone() for p in stiff.actor.parameters()]
    critic_before = [p.detach().clone() for p in stiff.critic.parameters()]
    stiff.current_horizon_lr = soft.current_horizon_lr = 1e-2

    infos = []
    for trainer in (stiff, soft):
        rollout, _ = trainer.rollout()
        info = trainer.update_actor(rollout)
        info.update(trainer.update_critic(rollout))
        info.update(trainer.update_horizon(rollout))
        infos.append(info)

    assert all(torch.isfinite(torch.tensor(v)) for info in infos for v in info.values())
    assert any(not torch.equal(a, p) for a, p in zip(actor_before, stiff.actor.parameters()))
    assert any(not torch.equal(a, p) for a, p in zip(critic_before, stiff.critic.parameters()))
    assert 1 <= infos[0]["update/critic_epochs"] <= 64
    # Reference sign: contact above the threshold lengthens the horizon, below it shortens (clipped at 4).
    assert stiff.horizon > 4.0 and len(stiff.dual) == round(stiff.horizon)
    assert soft.horizon == 4.0 and len(soft.dual) == 4


def test_learning_rates_decay_linearly():
    trainer = make_trainer(PointMassEnv())

    trainer.set_learning_rates(5)

    assert trainer.actor_optimizer.param_groups[0]["lr"] == pytest.approx((1e-5 + 2e-3) / 2)
    assert trainer.critic_optimizer.param_groups[0]["lr"] == pytest.approx((1e-5 + 5e-4) / 2)
    assert trainer.current_horizon_lr == pytest.approx((1e-5 + 5e-4) / 2)


def test_training_improves_point_mass_return():
    torch.manual_seed(0)
    env = PointMassEnv(num_envs=16, episode_length=1000)
    trainer = make_trainer(env, scale=1.0)
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
    trainer = Trainer("Ant", 4, seed=0, min_horizon=4, max_horizon=8, max_epochs=1, device="cuda")

    rollout, _ = trainer.rollout()
    info = trainer.update_actor(rollout)
    info.update(trainer.update_critic(rollout))
    info.update(trainer.update_horizon(rollout))

    assert rollout["rewards"].shape == rollout["contact_norms"].shape == (4, 4)
    assert info["update/actor_grad_norm"] > 0
    assert all(torch.isfinite(torch.tensor(v)) for v in info.values())
