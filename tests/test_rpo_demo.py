import pytest
import torch

from RLAlg.alg.rpo import RPO
from RPO.train import Trainer
from test_shac_demo import PointMassEnv


def make_trainer(env, **options):
    return Trainer("PointMass", env.num_envs, seed=0, horizon=4, max_epochs=10, device="cpu", env=env, **options)


def test_rollout_keeps_action_graph_and_bootstraps_from_critic_mean():
    env = PointMassEnv(episode_length=3)
    trainer = make_trainer(env)
    seen_mean = []
    forward = trainer.critic.forward

    def record(x):
        steps = forward(x)
        seen_mean.append(torch.stack([step.value for step in steps]).detach().mean(0))
        return steps

    trainer.critic.forward = record
    rollout, episode_info = trainer.rollout()

    assert len(rollout["actions"]) == 4 and all(a.requires_grad for a in rollout["actions"])
    for key in ("rewards", "entropies", "next_values", "terminated", "truncated"):
        assert rollout[key].shape == (4, env.num_envs)
    torch.testing.assert_close(rollout["next_values"].detach(), torch.stack(seen_mean))
    assert episode_info["episode/mean_length"] == 3.0
    assert trainer.target_entropy == -0.5 * env.num_act


def test_first_policy_epoch_is_on_policy_and_later_epochs_move_away(monkeypatch):
    trainer = make_trainer(PointMassEnv(terminate_below=1.2))
    actor_before = [p.detach().clone() for p in trainer.actor.parameters()]
    critic_before = [p.detach().clone() for p in trainer.critic.parameters()]
    clip_fractions, kls = [], []
    compute = RPO.compute_policy_loss

    def record(*args, **kwargs):
        result = compute(*args, **kwargs)
        clip_fractions.append(result["policy_clip_fraction"].item())
        kls.append(result["kl_divergence"].item())
        return result

    monkeypatch.setattr(RPO, "compute_policy_loss", staticmethod(record))
    rollout, _ = trainer.rollout()
    info = trainer.update_actor(rollout)
    info.update(trainer.update_critic(rollout))

    assert len(kls) == trainer.actor_iterations
    assert clip_fractions[0] == 0.0 and kls[0] < 1e-6 < kls[-1]
    assert info["update/skipped"] == 0.0
    assert all(torch.isfinite(torch.tensor(v)) for v in info.values())
    assert any(not torch.equal(a, p) for a, p in zip(actor_before, trainer.actor.parameters()))
    assert any(not torch.equal(a, p) for a, p in zip(critic_before, trainer.critic.parameters()))


def test_non_finite_action_gradients_skip_the_policy_update():
    trainer = make_trainer(PointMassEnv())
    actor_before = [p.detach().clone() for p in trainer.actor.parameters()]
    rollout, _ = trainer.rollout()
    rollout["rewards"] = rollout["rewards"] * float("nan")

    info = trainer.update_actor(rollout)

    assert info["update/skipped"] == 1.0
    assert all(torch.equal(a, p) for a, p in zip(actor_before, trainer.actor.parameters()))


def test_learning_rates_decay_exponentially_to_min_lr():
    trainer = make_trainer(PointMassEnv())

    trainer.set_learning_rates(5)
    assert trainer.actor_optimizer.param_groups[0]["lr"] == pytest.approx((5e-4 * 1e-5) ** 0.5)
    trainer.set_learning_rates(10)
    assert trainer.critic_optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)


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

    assert rollout["rewards"].shape == rollout["entropies"].shape == (4, 4)
    assert info["update/skipped"] == 0.0 and info["update/actor_grad_norm"] > 0
    assert all(torch.isfinite(torch.tensor(v)) for v in info.values())
