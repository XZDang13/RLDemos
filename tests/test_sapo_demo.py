import pytest
import torch

from SAPO.train import Trainer
from test_shac_demo import PointMassEnv


def make_trainer(env, **options):
    return Trainer("PointMass", env.num_envs, seed=0, horizon=4, max_epochs=10, device="cpu", env=env, **options)


def test_rollout_records_entropies_and_both_critic_values():
    env = PointMassEnv(episode_length=3)
    trainer = make_trainer(env)

    rollout, episode_info = trainer.rollout()

    for key in ("rewards", "entropies", "next_values", "min_next_values", "terminated", "truncated"):
        assert rollout[key].shape == (4, env.num_envs)
    assert rollout["rewards"].requires_grad and rollout["entropies"].requires_grad
    assert rollout["next_values"].requires_grad and not rollout["min_next_values"].requires_grad
    assert (rollout["min_next_values"] <= rollout["next_values"].detach() + 1e-6).all()
    assert episode_info["episode/mean_length"] == 3.0
    assert trainer.target_entropy == -0.5 * env.num_act


def test_next_values_are_mean_for_policy_and_min_for_targets():
    env = PointMassEnv(num_envs=2)
    trainer = make_trainer(env)
    seen_min, seen_mean = [], []
    forward = trainer.critic.forward

    def record(x):
        steps = forward(x)
        values = torch.stack([step.value for step in steps]).detach()
        seen_min.append(values.min(0).values)
        seen_mean.append(values.mean(0))
        return steps

    trainer.critic.forward = record
    rollout, _ = trainer.rollout()

    torch.testing.assert_close(rollout["next_values"].detach(), torch.stack(seen_mean))
    torch.testing.assert_close(rollout["min_next_values"], torch.stack(seen_min))


def test_update_changes_parameters_and_temperature():
    trainer = make_trainer(PointMassEnv(terminate_below=1.2))
    actor_before = [p.detach().clone() for p in trainer.actor.parameters()]
    critic_before = [p.detach().clone() for p in trainer.critic.parameters()]

    rollout, _ = trainer.rollout()
    info = trainer.update_actor(rollout)
    info.update(trainer.update_alpha(rollout))
    info.update(trainer.update_critic(rollout))

    assert all(torch.isfinite(torch.tensor(v)) for v in info.values())
    assert any(not torch.equal(a, p) for a, p in zip(actor_before, trainer.actor.parameters()))
    assert any(not torch.equal(a, p) for a, p in zip(critic_before, trainer.critic.parameters()))
    # Entropy above the target lowers the temperature and below it raises it.
    above = rollout["entropies"].mean().item() > trainer.target_entropy
    assert (info["update/alpha"] < 1.0) == above


def test_learning_rates_decay_linearly_except_alpha():
    trainer = make_trainer(PointMassEnv())

    trainer.set_learning_rates(5)

    assert trainer.actor_optimizer.param_groups[0]["lr"] == pytest.approx((1e-5 + 2e-3) / 2)
    assert trainer.critic_optimizer.param_groups[0]["lr"] == pytest.approx((1e-5 + 5e-4) / 2)
    assert trainer.alpha_optimizer.param_groups[0]["lr"] == pytest.approx(5e-3)


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
    info.update(trainer.update_alpha(rollout))
    info.update(trainer.update_critic(rollout))

    assert rollout["rewards"].shape == rollout["entropies"].shape == (4, 4)
    assert info["update/actor_grad_norm"] > 0
    assert all(torch.isfinite(torch.tensor(v)) for v in info.values())
