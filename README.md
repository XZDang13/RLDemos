# RLDemos

**English** | [简体中文](README.zh-CN.md)

Reinforcement learning examples built on the neighboring `../RLAlg` repository.
They use the current RLAlg interfaces for `NormPosition`, loss dictionaries,
and explicit termination and timeout handling. Python 3.10 or later and an
RLAlg checkout containing these interfaces are required.

| Directory | Algorithm | Data source | Action space |
| --- | --- | --- | --- |
| `PPO` | Proximal Policy Optimization | Online rollouts | Discrete or Box |
| `DDPG` | Deep Deterministic Policy Gradient | Online replay buffer | Box |
| `DDPG_DQ` | RLAlg's DDPGDoubleQ | Online replay buffer | Box |
| `SAC` | Soft Actor-Critic | Online replay buffer | Box |
| `DSAC` | Distributional SAC | Online replay buffer, target actor and critic | Box |
| `DSACT` | Distributional SAC with twin critics | Online replay buffer, running standard deviation statistics | Box |
| `FPO` | Flow Policy Optimization | Online rollouts with stored CMF sampling targets | Box |
| `IQL` | Implicit Q-Learning | Fixed offline `.npz` dataset | Box |
| `SHAC` | Short-Horizon Actor-Critic | Differentiable Rewarped/DFlex rollouts | Box in `[-1, 1]` |
| `AHAC` | Adaptive Horizon Actor-Critic | Differentiable Rewarped/DFlex rollouts | Box in `[-1, 1]` |
| `SAPO` | Soft Analytic Policy Optimization | Differentiable Rewarped/DFlex rollouts | Box in `[-1, 1]` |

Box action spaces must be one-dimensional, finite, and symmetric around zero.
Observations are flattened with `FlattenObservation`. These examples use MLPs
and are intended for vector observations. Each `model.py` defines the networks,
and each `train.py` demonstrates the training loop.

## Installation

Run from the RLDemos root directory:

```bash
python -m pip install -r requirements.txt
python -m pip install -e ../RLAlg
```

PPO, DDPG, DDPG_DQ, SAC, DSAC, DSACT, and FPO default to `HalfCheetah-v5`.
This environment requires the optional MuJoCo dependencies:

```bash
python -m pip install 'gymnasium[mujoco]>=1.1,<2'
```

The short runs below use `Pendulum-v1` and `CartPole-v1`, which do not require
MuJoCo. All examples print metrics to the terminal by default. To use W&B,
install `wandb` and add `--wandb`.

## Online examples

```bash
python -m PPO.train --env CartPole-v1 --env-num 4 --epochs 10 --rollout-steps 128 --batch-size 128
python -m PPO.train --env Pendulum-v1 --env-num 4 --epochs 10 --rollout-steps 128 --batch-size 128
python -m DDPG.train --env Pendulum-v1 --env-num 4 --epochs 10 --buffer-steps 10000 --iterations 50
python -m DDPG_DQ.train --env Pendulum-v1 --env-num 4 --epochs 10 --buffer-steps 10000 --iterations 50
python -m SAC.train --env Pendulum-v1 --env-num 4 --epochs 10 --buffer-steps 10000 --iterations 50
python -m DSAC.train --env Pendulum-v1 --env-num 4 --epochs 10 --buffer-steps 10000 --iterations 50
python -m DSACT.train --env Pendulum-v1 --env-num 4 --epochs 10 --buffer-steps 10000 --iterations 50
python -m FPO.train --env Pendulum-v1 --env-num 4 --epochs 10 --rollout-steps 128 --batch-size 128
```

You can also run scripts directly, for example `python PPO/train.py ...`.
Use `--help` to see the available options:

- `--env`, `--env-num`, `--seed`: environment, number of parallel environments,
  and random seed.
- `--epochs`, `--iterations`, `--batch-size`: training epochs, updates per epoch,
  and batch size. Each PPO/FPO update iteration traverses the entire rollout;
  each update iteration in the other online algorithms samples one replay batch.
- `--rollout-steps`: steps collected per environment per epoch. Defaults to the
  environment's episode time limit.
- `--buffer-steps`: replay steps retained per environment for off-policy
  examples. Total capacity is `env-num × buffer-steps`.
- `--device cpu` / `--device cuda`: training device; selected automatically by default.
- `--wandb`: enable RLAlg's `WandbLogger`.

The PPO continuous policy learns `log_std` by default. The new interface also
supports learning the standard deviation directly:

```bash
python -m PPO.train --env Pendulum-v1 --std-parameterization std --init-std 0.6
```

PPO/FPO distinguish `terminated` from `truncated`. By default, timeouts bootstrap
from the actual final observation using `V(final_obs)` to adjust the reward,
and GAE stops at episode boundaries. Use `--no-bootstrap-timeouts` to disable
timeout bootstrapping. Since rewards already include this correction,
`compute_gae` is called without its optional `V(current_obs)` timeout correction.
Vector environments use explicit partial resets so reset steps are not stored
as transitions. Off-policy examples retain the actual `next_observations`
and use only `terminated` to mask Bellman bootstrapping.

For FPO, `--flow-steps` controls the number of Euler sampling steps and
`--cmf-samples` controls the number of CMF samples per action. Their defaults are
8 and 10, respectively. The replay buffer stores latent actions, noise, time,
and initial CMF losses, and updates reuse these sampling targets. Actions sent
to the environment are transformed with `tanh × max_action`.

FPO uses `exp(mean(old_loss - new_loss))` by default, giving one ratio per action
before PPO clipping. The log ratio is bounded after averaging to avoid
exponential overflow. Setting `average_losses_before_exp=False` in RLAlg
explicitly selects the per-sample ratio ablation. The default follows
[Algorithm 1 of the paper](https://arxiv.org/html/2507.21053v2#S3).

`--clip-ratio` defaults to 0.05. `--supervise-target noise` uses a noise
reconstruction loss, while `velocity` uses velocity MSE. The network predicts
velocity in both modes, with `t=0` representing the action and `t=1` the noise.
`--time-step-sampler continuous` is the default; `discrete` is also supported.
Discrete sampling uses the Euler step starting times. With only one step,
the `noise` loss is always zero at `t=1`, so this configuration is rejected.
Use continuous sampling or the `velocity` mode instead.

By default, sampling integrates random initial noise without adding extra
endpoint noise. The same initial noise produces the same action. Old losses,
noise/time pairs, and advantages remain fixed during updates. `approx_kl` is
a nonnegative drift metric based on the surrogate ratio, rather than an exact
KL divergence between flow policies. The legacy `kl_divergence` key remains
an alias for this metric.

FPO normalizes observations by default and freezes the statistics throughout
each rollout and all its updates. Statistics are updated from the previous
rollout only before the next rollout starts, so old and new CMF losses use the
same input transformation. Use `--no-normalize-observations` to disable this.
The default `--reward-scale 0.1` scales training rewards only; episode logs and
independent evaluation still report raw returns. Timeout values are already
in scaled units, so only environment rewards are scaled, without scaling the
bootstrap value again.

The value network uses unclipped MSE by default, avoiding the use of the
policy's 0.05 ratio range as an absolute limit on value changes.
Use `--value-clip-range` to set a separate value clipping range.
CPU training uses one compute thread by default; adjust this with `--threads`.

Independent evaluation uses separate environments and random numbers from
training. Each checkpoint uses the same four evaluation seeds and initial
noise sequence. The following experiments record raw returns, training metrics,
and policy parameter changes at every checkpoint:

```bash
python -m FPO.benchmark --env HalfCheetah-v5 --seed 0 --epochs 100 --output FPO/results/halfcheetah_seed0.json
python -m FPO.benchmark --env Pendulum-v1 --seed 0 --epochs 120 --rollout-steps 200 --output FPO/results/pendulum_seed0.json
```

Repeat with `--seed 1` and `--seed 2` to check learning across random seeds.
See the [FPO experiment report (Chinese)](FPO/results/REPORT.md) for numerical
results and comparison configurations.

## SHAC differentiable simulation example

SHAC backpropagates through the simulator, so it uses
[Rewarped](https://github.com/rewarped/rewarped) DFlex tasks instead of
Gymnasium. Rewarped 1.3.3 pins `warp-lang==1.3.3`, which still has the
`warp.sim` module it needs. It cannot share an environment with Isaac Lab or
other packages that require a newer Warp. Use a separate CUDA environment:

```bash
conda create -n rewarped python=3.10
conda activate rewarped
python -m pip install torch torchvision gym==0.23.1 rewarped
python -m pip install -r requirements.txt
python -m pip install -e ../RLAlg
```

The defaults follow the reference SHAC Ant configuration: 128 environments,
horizon 32, `[400, 200, 100]` ELU networks, actor/critic learning rates `2e-3`
and `5e-4` decayed linearly, Adam betas `(0.7, 0.95)`, TD(λ=0.95) targets with
16 critic iterations over 4 minibatches, a target critic with `tau=0.01`, and
gradient clipping at 1.0:

```bash
python -m SHAC.train --env Ant --epochs 2000
python -m SHAC.train --env Ant --env-num 16 --epochs 3 --horizon 8  # quick check
```

`--env` accepts `Ant`, `Hopper`, `Humanoid`, `Cartpole`, `Anymal`, and `A1`.
Each epoch starts with `env.initialize_trajectory()`, which keeps the simulator
state but cuts the graph to earlier epochs. Next values come from the target
critic on `obs_before_reset`, so terminated episodes do not bootstrap and
timeouts do. Observation statistics are frozen during a rollout and updated
afterwards. Actor updates with non-finite gradients are skipped.
`episode/mean_rewards` reports the undiscounted return of stochastic training
episodes.

A default Ant run with seed 0 (2000 epochs, 8.19M steps, about 68 minutes on an
RTX 3090) raised the mean training return from about 690 over the first 200
epochs to about 1500 over the last 200. Most episodes reached the 1000-step limit
after the first few hundred epochs.

`Trainer(..., env=...)` accepts any object with the same API. The tests use this
with a differentiable point-mass environment, so they run without Rewarped.

## AHAC differentiable simulation example

AHAC uses the same Rewarped environment and rollout loop as SHAC (install it as
above), with three changes from the
[reference implementation](https://github.com/imgeorgiev/DiffRL):

- The rollout horizon adapts between 8 and 64 steps. After each epoch,
  `AHAC.update_horizon` moves dual variables by how far the per-step contact
  stiffness sits above or below `C = 500` and moves the horizon by their sum
  with a learning rate of `5e-4`, decayed linearly to `1e-5`. As in the
  reference, contact above the threshold lengthens the horizon.
- Next values come from `min(V1, V2)` of a live double critic; there is no
  target critic.
- The critic trains on 4 minibatches per epoch until the epoch loss stops
  changing (`AHAC.critic_converged`), for at most 64 epochs. Non-finite critic
  gradients are zeroed, as in the reference.

The other defaults match the SHAC demo: 128 environments, `[400, 200, 100]` ELU
networks, actor/critic learning rates `2e-3` and `5e-4`, Adam betas `(0.7, 0.95)`,
TD(λ=0.95) targets, and actor gradient clipping at 1.0.

```bash
python -m AHAC.train --env Ant --epochs 2000
python -m AHAC.train --env Ant --env-num 16 --epochs 3  # quick check
```

Rewarped does not report contact forces, so [contact.py](AHAC/contact.py) wraps
the Featherstone integrator's `simulate` and reads the last substep's rigid
contact forces (`body_f`) and body accelerations (`body_a_s`). It keeps state
references, so it also works with CUDA graph capture. The contact norm is
`||forces / max(accelerations, 1)||` over all bodies. Only the linear parts are
used, because Warp expresses wrenches about the world origin and Rewarped
spaces environments apart, which would make the torque parts depend on an
environment's position. The probe requires the Featherstone integrator, which
all six DFlex tasks use.
The horizon and dual variables are saved in the checkpoint.

A default Ant run with seed 0 (2000 epochs, 14.6M steps, about 137 minutes on an
RTX 3090) raised the mean training return from about 1170 over the first 200
logged epochs to about 1240 over the last 200, with a peak of about 2200. On
this seed it ended below the SHAC demo's 1500. The horizon stayed at 8 while the
contact norm was below the threshold, then grew once the gait made stiffer
contact, reaching the 64-step maximum after about 0.7M steps and staying there.
Epochs at the full horizon take about 4-5 seconds.

`Trainer(..., env=..., contact_probe=...)` accepts any environment with the
SHAC demo's API and a callable that returns `(contact_forces, accelerations)`.
The tests use the point-mass environment with a synthetic probe.

## SAPO differentiable simulation example

SAPO uses the same Rewarped environment and rollout loop as SHAC (install it as
above). Its settings follow the
[reference DFlex Ant config](https://github.com/etaoxing/mineral):

- Every reward gains `alpha * normalized_entropy` in both the policy return and
  the TD(λ) targets. The entropy is the one-sample estimate `-log_prob` of the
  tanh-squashed action.
- After each actor step, `alpha` (initially 1) moves toward the target entropy
  `-0.5 * action_dim` with AdamW at a constant learning rate of `5e-3`. The
  critic targets use the updated `alpha`.
- Two live critics, with no target critic. The policy bootstraps from their mean,
  the targets from their minimum.
- 64 environments, horizon 32, and 5000 epochs (10.2M steps). The actor is a
  LayerNorm SiLU `[128, 64, 32]` network with state-dependent log-std in `[-5, 2]`;
  each critic is `[64, 64]`. AdamW with betas `(0.7, 0.95)` and learning rates
  `2e-3` (actor) and `5e-4` (critic), both decayed linearly to `1e-5`. Gradient
  norms are clipped at 0.5, and the critic trains for a fixed 16 epochs of 4
  minibatches. Non-finite critic gradients are zeroed, as in the reference.

```bash
python -m SAPO.train --env Ant
python -m SAPO.train --env Ant --env-num 16 --epochs 3  # quick check
```

The temperature and its optimizer are saved in the checkpoint.

A default Ant run with seed 0 (5000 epochs, 10.2M steps, about 123 minutes on an
RTX 3090) raised the mean training return from about 1220 over the first 200
logged epochs to about 4780 over the last 200, with a peak of about 5480. That
is well above the SHAC (about 1500) and AHAC (about 1240) demos, though those
used fewer steps (8.2M and 14.6M) and different networks. The temperature fell
from 1 to about 0.27 early on, then settled near 0.4-0.5 as the entropy held
near its target of -4.

## IQL offline example

Collect a dataset with a random policy, then train from the saved file:

```bash
python -m IQL.collect --env Pendulum-v1 --steps 10000 --output data/pendulum_random.npz
python -m IQL.train --env Pendulum-v1 --dataset data/pendulum_random.npz --epochs 10 --iterations 100 --eval-episodes 5
```

The random dataset demonstrates the workflow; policy performance depends on
offline data quality. Training reads a fixed dataset without collecting or
appending transitions from the environment. `--eval-episodes` separately enables
evaluation after training. `--expectile` and `--temperature` default to 0.7 and
3.0, respectively. The collector refuses to overwrite an existing file unless
`--overwrite` is supplied.

You can provide your own `.npz` dataset with the following fields:

| Key | Shape | Meaning |
| --- | --- | --- |
| `observations` | `[N, obs_dim]` | Flattened current observations |
| `next_observations` | `[N, obs_dim]` | Actual next observations, including episode final states |
| `actions` | `[N, action_dim]` | Actions within the environment's bounds |
| `rewards` | `[N]` | Rewards |
| `terminated` | `[N]` | True termination flags, bool or 0/1; must not include timeouts |
| `truncated` | `[N]` | Optional timeout flags; defaults to False |
| `env_name` | Scalar string | Optional environment ID for checking dataset/environment compatibility |

## Checkpoints

Every `train.py` saves a checkpoint after training finishes, by default to
`checkpoints/<algorithm>-<env>-<seed>.pt` (ignored by Git). Use `--save-path`
to choose another location. The demos without a command line (AMP, GAIL, HER,
PPO_RNN, PPO_split, SAC_Auto, SAC_asym) always use the default path.
A checkpoint stores the `state_dict` of every network, normalizer and
optimizer on the trainer, plus free learnable tensors such as `log_alpha`.
DSACT also saves its running critic standard-deviation statistics, which a
trainer lists in `extra_checkpoint_state`. Environments, replay buffers and
loggers are not saved. The metadata records
the algorithm, environment, seed and `global_step`. To restore, build a trainer
with the same options and load the state into it:

```python
from demo_utils import load_checkpoint
from SAC.train import Trainer

trainer = Trainer("Pendulum-v1", 4)
metadata = load_checkpoint(trainer, "checkpoints/SAC-Pendulum-v1-0.pt")
```

## Validation

Quickly check one rollout and update for each online algorithm:

```bash
python -m PPO.train --env CartPole-v1 --env-num 2 --epochs 1 --iterations 1 --rollout-steps 8 --batch-size 8 --device cpu
python -m FPO.train --env Pendulum-v1 --env-num 2 --epochs 1 --iterations 1 --rollout-steps 8 --batch-size 8 --flow-steps 2 --cmf-samples 3 --device cpu
for algorithm in DDPG DDPG_DQ SAC DSAC DSACT; do
    python -m "$algorithm.train" --env Pendulum-v1 --env-num 2 --epochs 1 --iterations 1 --rollout-steps 8 --buffer-steps 32 --batch-size 8 --device cpu
done
```

Regression tests cover parameter updates, discrete action types, termination
and timeouts, actual final observations, partial resets when environments finish
at different times, and DDPGDoubleQ noise arguments. FPO checks cover reuse of
CMF targets, frozen observation statistics, reward/timeout scaling, value
gradients, evaluation randomness, and reward learning across three random seeds.
IQL checks cover offline training and dataset validation. Checkpoint tests cover
save/load round trips for every maintained trainer and the `--save-path` option:

```bash
python -m pip install pytest
python -m pytest -q
```

`test.ipynb` provides an interactive example of vector environments, episode
statistics, and partial resets.
