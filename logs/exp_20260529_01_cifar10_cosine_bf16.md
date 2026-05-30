# Experiment Log 

`exp_20260529_01_cifar10_cosine_bf16`

## 一句话目标

> 本次实验最想验证什么？答案应当是 yes/no 形式可判定的。

*验证在 CIFAR-10 数据集上使用 Cosine Schedule 时，将混合精度从 fp16 切换为 bf16 能否消除由于数值溢出导致的“纯色块”现象，并最终将 EMA 模型的 FID 分数成功压低至 15 以内。*

---

## 1. 环境信息

| 项目 | 内容 |
|------|------|
| 实验日期 | 2026-05-29 |
| 实验者 | HZQ |
| 硬件 | NVIDIA GeForce RTX 3060 6GB × 1 |
| 框架版本 | PyTorch 2.3.0, CUDA 12.1 |
| 仓库 | github.com/coding-all-the-time/diffusion-model-prectical-project1 |
| Git commit | `待填` |
| Git 分支 | `origin/main` |

---

## 2. 假设与预期

> 在跑实验**之前**写下你的预期。之后对比结果，培养科学直觉。

**假设**：
- 前一次实验在中后期（约 45000 步）采样发生崩溃、生成大量纯色块，是因为 fp16 动态范围过窄（最大仅支持 ~65504）。Cosine 调度使得高噪声状态持续时间变长，U-Net 内部的 Attention 层容易产生极大的激活值或梯度，导致 NaN/Inf 溢出。
- bf16 使用与 fp32 相同的 8 位指数位（动态范围高达 $\sim 10^{38}$），仅牺牲了尾数精度。切换到 bf16 后，能在显存占用基本不变的情况下，彻底根除数值溢出导致的权重崩坏问题。
- 在网络不再崩坏的前提下，Cosine schedule 应当能发挥其在低噪声区间的细粒度去噪优势，取得比 Linear 更好的生成细节。

**预期结果**：
- 显存占用依然保持在 4.0 GB 左右（与 fp16 相当）。
- 训练总时长在 15 小时左右。
- 45000 步之后，依然能稳定生成含有语义的图像，不会再出现突兀的纯白、纯黑、纯黄等纯色块。
- 最终生成的图像在颜色和结构上比 Linear baseline 更合理，EMA 模型的 FID 分数突破 15。

**如果与预期不符可能的原因**：
- 如果依然出现纯色块，可能不是由于精度溢出导致的，而是由于极端的学习率 (2.0e-4) 在 Cosine 调度的后期显得过大，导致模型在极小梯度的情况下发散。
- 如果 FID 依然不如 Linear baseline，说明对于当前配置（128 base_channels, 200 epoch），模型容量或训练步数还没达到 Cosine Schedule 所需的收敛阈值。

---

## 3. 配置（完整 hyperparameters）

```yaml
# CIFAR-10 DDPM Configuration（Cosine 进阶档）
# 期望训练时长: ~15h on RTX 3060 6GB
# 期望结果: 200 epoch 后 FID ≤ 15 (5000 samples)

output_dir: runs/exp_cifar10_cosine_bf16
seed: 42

# ─────────── 数据 ───────────
dataset:
  name: cifar10
  batch_size: 64 
  image_size: 32
  num_workers: 4
  root: ./data

# ─────────── 模型（标准 U-Net） ───────────
model:
  in_channels: 3
  out_channels: 3
  base_channels: 128         # CIFAR-10 标配
  channel_mult: [1, 2, 2, 2] # 32 → 16 → 8 → 4 → 4
  num_res_blocks: 2
  attn_resolutions: [16]      # 16x16 加 self-attention
  dropout: 0.1
  image_size: 32

# ─────────── Diffusion ───────────
diffusion:
  T: 1000
  beta_schedule: cosine 
  s: 0.008 

# ─────────── Optimizer ───────────
optimizer:
  lr: 2.0e-4
  weight_decay: 0.0
  warmup_steps: 5000

# ─────────── EMA ───────────
ema_decay: 0.9999

# ─────────── 混合精度 ───────────
mixed_precision: bf16 

# ─────────── 训练循环 ───────────
training:
  num_epochs: 200
  log_every: 100
  sample_every: 2500
  ckpt_every: 10000
  grad_clip: 1.0
  max_keep_ckpts: 3           # 最多保留最近的 3 个 checkpoint 节省硬盘

# ─────────── WandB ───────────
wandb:
  enabled: true             
  project: ddpm-course
  run_name: cifar10_cosine_bf16  

# ─────────── 绘图（可选） ───────────
plotting:
  enabled: true             
  record_every: 50          
  plot_every: 1000
```

---

## 4. 运行命令（必须可直接复制粘贴运行）

```bash
python train.py --config configs/cifar10_cosine_bf16.yaml
```
如果中途中断，使用以下命令恢复
```bash
python train.py --config configs/cifar10_cosine_bf16.yaml --resume runs/exp_cifar10_cosine_bf16/ckpt/step_XXXXXX.pt # 替换为具体 checkpoint名
```

---

## 5. 结果

### 5.1 数值指标

| 指标 | 数值 | 备注 |
|------|------|------|
| 训练总时长 | 14 h 56 min |使用140W便携适配器供电 |
| 显存峰值 | 3.71 GB | |
| 最终训练 loss | 0.06508 | |
| FID @ EMA| - | 5000 张样本 |
| FID @ no EMA| - | 5000 张样本 |

### 5.2 可视化结果

`wandb/run-20260529_200237-t6x62lju`

[wandb.ai/worldexplorer111111-tsinghua-university/ddpm-course/runs/t6x62lju](https://wandb.ai/worldexplorer111111-tsinghua-university/ddpm-course/runs/t6x62lju)

（注册时以为能改，选了清华，后面发现改不了了...）

### 5.3 与 baseline 对比

| 配置 | FID(ema/no_ema) | 训练时长 | 备注 |
|------|-----|----------|------|
| linear schedule/fp16 (baseline) | 17.8579/33.0284 | 14 h 53 min |exp_20260527_01，使用140W便携适配器供电 |
| cosine schedule/fp16 (上次失败) | 62.0114/189.3443 | 15 h 11 min |exp_20260528_01，使用140W便携适配器供电 |
| cosine schedule/bf16 (本次) | -/- | 14 h 56 min |exp_20260529_01，使用140W便携适配器供电 |

---

## 6. 结论

### 6.1 假设是否成立

❌ 不成立

**简短说明**：
1. bf16 每采样步采出来的图片和 fp16 几乎一样，不仅依然出现了纯色块，而且出现的时机都高度重合。证明纯色块问题不是因为半精度浮点数的动态范围不够导致。
2. 可能是 Cosine Schedule 在 $t \to T$ 时的特性：其极低的信噪比导致采样推断初始阶段的微小预测误差被瞬间放大。如果没有显式限制预测值域，这种误差会导致预测出的原始图像 $x_0$ 发生数值爆炸，最终被 denormalize 截断成毫无特征的纯色输出。

### 6.2 现象记录

观察到但与预期无关的现象：

- 使用 bf16 时，GPU 的功率限制从 50W 变成 95W ，电脑耗电变快(充着电电量还在掉)，训练速度加快(电池充足时为 3.5 step/s，电池不足时为 2.9 step/s)。

---

## 7. 失败记录与踩坑（**强制填写**）

> 没有遇到任何问题是不正常的。仔细回想是否有任何"花了 5 分钟以上才解决的事情"。

### 坑 1：

---

## 8. 下一步计划

> 基于本次结果，下一个实验应该做什么？写得越具体越好。

- [ ] 修改 diffusion.py：在 p_sample中，当利用模型输出的 $\epsilon$ 推算当前的 $x_0$ 时，加上 pred_x0.clamp_(-1.0, 1.0)。
- [ ] 将 optimizer.lr 从 2.0e-4 下调至 1.5e-4。


---

## 9. 给未来自己的提醒

> 一句话总结这次实验最值得记住的事情。

**

---
