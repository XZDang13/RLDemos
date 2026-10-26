# RLDemos

[English](README.md) | **简体中文**

基于相邻目录 `../RLAlg` 的强化学习例子，适配当前 RLAlg 的
`NormPosition`、字典形式的损失返回值，以及显式终止/超时处理接口。
需要 Python 3.10 或更高版本，以及包含这些接口的最新 RLAlg 源码。

| 目录 | 算法 | 数据来源 | 动作空间 |
| --- | --- | --- | --- |
| `PPO` | Proximal Policy Optimization | 在线 rollout | Discrete 或 Box |
| `DDPG` | Deep Deterministic Policy Gradient | 在线 replay buffer | Box |
| `DDPG_DQ` | RLAlg 的 DDPGDoubleQ | 在线 replay buffer | Box |
| `SAC` | Soft Actor-Critic | 在线 replay buffer | Box |
| `DSAC` | Distributional SAC | 在线 replay buffer，目标 actor/critic | Box |
| `DSACT` | Distributional SAC with twin critics | 在线 replay buffer，运行标准差统计 | Box |
| `FPO` | Flow Policy Optimization | 在线 rollout，保存 CMF 采样目标 | Box |
| `IQL` | Implicit Q-Learning | 固定离线 `.npz` 数据集 | Box |
| `SHAC` | Short-Horizon Actor-Critic | 可微 Rewarped/DFlex rollout | `[-1, 1]` 内的 Box |

Box 动作空间目前要求一维、有限且关于 0 对称。观测通过 `FlattenObservation`
展平；例子使用 MLP，不面向像素观测。各 `model.py` 展示网络，`train.py` 展示训练流程。

## 安装

在 RLDemos 根目录执行：

```bash
python -m pip install -r requirements.txt
python -m pip install -e ../RLAlg
```

PPO、DDPG、DDPG_DQ、SAC、DSAC、DSACT、FPO 保留默认的 `HalfCheetah-v5` 环境。
运行此环境需要额外安装 MuJoCo：

```bash
python -m pip install 'gymnasium[mujoco]>=1.1,<2'
```

下面的短训练使用 `Pendulum-v1` 和 `CartPole-v1`，无需 MuJoCo。
所有例子默认将指标输出到终端；若需要 W&B，安装 `wandb` 并添加 `--wandb`。

## 在线例子

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

也可以直接执行 `python PPO/train.py ...` 等脚本。使用 `--help` 查看参数：

- `--env`、`--env-num`、`--seed`：环境、并行数量和随机种子。
- `--epochs`、`--iterations`、`--batch-size`：训练轮数、每轮更新次数和 batch 大小。
  PPO/FPO 每次更新遍历整段 rollout；其他在线算法每次更新采样一个 replay batch。
- `--rollout-steps`：每个环境每轮采集的步数，默认使用环境的 episode 时间上限。
- `--buffer-steps`：每个环境保留的 replay 步数，适用于 off-policy 例子。
  总容量为 `env-num × buffer-steps`。
- `--device cpu` / `--device cuda`：指定训练设备；默认自动选择。
- `--wandb`：启用 RLAlg 的 `WandbLogger`。

PPO 连续策略默认学习 `log_std`，可以使用新接口直接学习标准差：

```bash
python -m PPO.train --env Pendulum-v1 --std-parameterization std --init-std 0.6
```

PPO/FPO 区分 `terminated` 与 `truncated`。超时时默认使用真实末状态的
`V(final_obs)` 修正 reward，并在 GAE 中切断 episode 边界；
`--no-bootstrap-timeouts` 关闭超时 bootstrap。由于已修正 reward，调用
`compute_gae` 时不再叠加其可选的 `V(current_obs)` 超时修正。
向量环境使用显式部分重置，避免将重置步记录为 transition。
Off-policy 例子保留真实 `next_observations`，只用 `terminated` 屏蔽 Bellman bootstrap。

FPO 的 `--flow-steps` 控制 Euler 采样步数，`--cmf-samples` 控制每个动作的 CMF
采样数量，默认分别为 8 和 10。ReplayBuffer 保存 latent action、噪声、时间和初始
CMF loss，更新时使用相同采样目标；发送给环境的动作通过 `tanh × max_action` 变换。

FPO 默认使用 `exp(mean(old_loss - new_loss))`，每个动作只有一个 ratio，
再进行 PPO clipping；在平均之后对 log ratio 限幅，避免指数溢出。
`average_losses_before_exp=False` 仍可在 RLAlg 中显式选择逐样本 ratio 的消融版本。
这个默认值按[论文算法 1](https://arxiv.org/html/2507.21053v2#S3)修正。

`--clip-ratio` 默认是 0.05。`--supervise-target noise` 使用噪声重建损失，
`velocity` 使用速度 MSE；两种模式的网络始终输出速度，且约定 `t=0` 为动作、
`t=1` 为噪声。`--time-step-sampler continuous` 默认连续采样，也支持 `discrete`。
离散采样使用 Euler 起点网格；只有一步时 `noise` 模式在 `t=1` 的损失恒为零，
因此此配置会被拒绝，可以改用连续采样或 `velocity` 模式。

采样默认只从随机初始噪声积分，不额外添加末端噪声；给定相同初始噪声可复现同一动作。
旧损失、噪声/时间和优势在更新时作为固定目标。`approx_kl` 是基于 surrogate ratio
的非负漂移指标，不是流策略的精确 KL；旧键 `kl_divergence` 保留为它的别名。

FPO 默认归一化观测，并在一段 rollout 及其所有更新中冻结统计量；只在下一段
rollout 开始前用上一段观测更新统计，保证旧、新 CMF 损失使用同一套输入变换。
`--no-normalize-observations` 可关闭归一化。默认 `--reward-scale 0.1` 只缩放训练
奖励，episode 日志和独立评估仍显示原始回报。超时修正中的价值已经是缩放后的
单位，因此只缩放环境奖励，不再次缩放 bootstrap 值。

价值网络默认使用未裁剪的 MSE，避免把策略的 0.05 比率范围误用为价值变化的
绝对上限。可用 `--value-clip-range` 单独指定价值裁剪范围。CPU 训练默认使用一个
计算线程，可以通过 `--threads` 调整。

独立评估与训练使用不同环境和随机数，每个检查点使用相同的 4 个评估种子和
初始噪声序列。以下实验保留每个检查点的原始回报、训练指标和策略参数变化：

```bash
python -m FPO.benchmark --env HalfCheetah-v5 --seed 0 --epochs 100 --output FPO/results/halfcheetah_seed0.json
python -m FPO.benchmark --env Pendulum-v1 --seed 0 --epochs 120 --rollout-steps 200 --output FPO/results/pendulum_seed0.json
```

重复使用 `--seed 1` 和 `--seed 2` 可检查不同随机种子的学习情况。
数值结果及修复前的对照设置见 [FPO 实验报告](FPO/results/REPORT.md)。

## SHAC 可微仿真例子

SHAC 需要对仿真器反向传播，因此使用 [Rewarped](https://github.com/rewarped/rewarped)
的 DFlex 任务，而不是 Gymnasium。Rewarped 1.3.3 锁定 `warp-lang==1.3.3`，
这个版本仍带有它依赖的 `warp.sim` 模块，所以不能和 Isaac Lab 等需要新版 Warp
的包共用环境。请使用独立的 CUDA 环境：

```bash
conda create -n rewarped python=3.10
conda activate rewarped
python -m pip install torch torchvision gym==0.23.1 rewarped
python -m pip install -r requirements.txt
python -m pip install -e ../RLAlg
```

默认超参数沿用官方 SHAC 的 Ant 配置：128 个环境、视界 32、`[400, 200, 100]`
的 ELU 网络、actor/critic 学习率 `2e-3` 和 `5e-4` 并线性衰减、Adam betas
`(0.7, 0.95)`、TD(λ=0.95) 目标、每轮 critic 迭代 16 次（4 个 minibatch）、
`tau=0.01` 的目标 critic，以及 1.0 的梯度裁剪：

```bash
python -m SHAC.train --env Ant --epochs 2000
python -m SHAC.train --env Ant --env-num 16 --epochs 3 --horizon 8  # 快速检查
```

`--env` 可选 `Ant`、`Hopper`、`Humanoid`、`Cartpole`、`Anymal`、`A1`。
每个 epoch 先调用 `env.initialize_trajectory()`：保留仿真状态，但截断到之前 epoch 的计算图。
下一状态价值由目标 critic 在 `obs_before_reset` 上计算，因此终止的 episode 不做 bootstrap，超时会做。
观测统计量在 rollout 期间冻结，rollout 结束后再更新。梯度出现非有限值时跳过该次 actor 更新。
`episode/mean_rewards` 是随机策略训练 episode 的未折扣回报。

用默认配置、seed 0 跑 Ant（2000 epochs，8.19M 步，RTX 3090 上约 68 分钟）：
前 200 个 epoch 的平均训练回报约 690，最后 200 个约 1500。训练几百个 epoch 后，
大多数 episode 都能跑满 1000 步上限。

`Trainer(..., env=...)` 可以接收任何接口相同的环境对象。测试用它接入一个可微的点质量环境，
因此不装 Rewarped 也能运行。

## IQL 离线例子

先采集一个随机策略数据集，再从文件训练：

```bash
python -m IQL.collect --env Pendulum-v1 --steps 10000 --output data/pendulum_random.npz
python -m IQL.train --env Pendulum-v1 --dataset data/pendulum_random.npz --epochs 10 --iterations 100 --eval-episodes 5
```

随机数据集用于演示流程；策略效果取决于离线数据质量。训练只读取固定数据，
不会向环境采集或追加 transition。`--eval-episodes` 单独启用训练后的评估。
`--expectile` 和 `--temperature` 默认分别为 0.7 和 3.0。
采集器默认拒绝覆盖已有文件，可显式添加 `--overwrite`。

可替换为自己的 `.npz` 数据集，需包含：

| 键 | 形状 | 含义 |
| --- | --- | --- |
| `observations` | `[N, obs_dim]` | 展平的当前观测 |
| `next_observations` | `[N, obs_dim]` | 实际下一观测，包括 episode 的末状态 |
| `actions` | `[N, action_dim]` | 环境范围内的动作 |
| `rewards` | `[N]` | reward |
| `terminated` | `[N]` | 真正终止，bool 或 0/1；不应合并超时 |
| `truncated` | `[N]` | 可选的超时标记，缺省为 False |
| `env_name` | 标量字符串 | 可选的环境 ID，用于检查数据与环境一致性 |

## Checkpoint

每个 `train.py` 训练结束后都会保存 checkpoint，默认路径是
`checkpoints/<算法>-<环境>-<seed>.pt`（已加入 `.gitignore`），可以用 `--save-path`
指定其他位置。没有命令行参数的例子（AMP、GAIL、HER、PPO_RNN、PPO_split、
SAC_Auto、SAC_asym）固定保存到默认路径。checkpoint 包含 trainer 上所有网络、
归一化器和优化器的 `state_dict`，以及 `log_alpha` 这类独立的可学习张量；
DSACT 还会保存 critic 标准差的滑动平均（trainer 在 `extra_checkpoint_state` 里列出）。
环境、replay buffer 和 logger 不保存。元数据记录算法、环境、seed 和 `global_step`。
恢复时用相同配置构造 trainer，再把状态加载进去：

```python
from demo_utils import load_checkpoint
from SAC.train import Trainer

trainer = Trainer("Pendulum-v1", 4)
metadata = load_checkpoint(trainer, "checkpoints/SAC-Pendulum-v1-0.pt")
```

## 验证

快速检查每个在线算法的一次采样和更新：

```bash
python -m PPO.train --env CartPole-v1 --env-num 2 --epochs 1 --iterations 1 --rollout-steps 8 --batch-size 8 --device cpu
python -m FPO.train --env Pendulum-v1 --env-num 2 --epochs 1 --iterations 1 --rollout-steps 8 --batch-size 8 --flow-steps 2 --cmf-samples 3 --device cpu
for algorithm in DDPG DDPG_DQ SAC DSAC DSACT; do
    python -m "$algorithm.train" --env Pendulum-v1 --env-num 2 --epochs 1 --iterations 1 --rollout-steps 8 --buffer-steps 32 --batch-size 8 --device cpu
done
```

回归测试覆盖参数更新、离散动作类型、终止/超时、真实末状态、异步部分重置、
DDPGDoubleQ 的噪声参数、FPO 的 CMF 目标复用、观测统计冻结、奖励/超时缩放、
价值梯度、独立评估随机数和三个随机种子的奖励学习检查，
IQL 的离线训练与数据校验，
以及所有维护中的 trainer 的 checkpoint 保存/加载往返和 `--save-path` 参数：

```bash
python -m pip install pytest
python -m pytest -q
```

`test.ipynb` 提供一个可交互执行的向量环境、episode 统计和部分重置示例。
