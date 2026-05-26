# Experiment Log 
## Experiment 1

`exp_20260525_01_mnist_baseline_2`

## 一句话目标

> 本次实验最想验证什么？答案应当是 yes/no 形式可判定的。

*验证fp16对于训练速度是否有提升，验证 EMA 衰减率改为 0.999对于训练质量是否有提升*

---

## 1. 环境信息

| 项目 | 内容 |
|------|------|
| 实验日期 | 2026-05-25 |
| 实验者 | HZQ |
| 硬件 | NVIDIA GeForce RTX 3060 6GB × 1 |
| 框架版本 | PyTorch 2.3.0, CUDA 12.1 |
| 仓库 | github.com/coding-all-the-time/diffusion-model-prectical-project1 |
| Git commit | `876fd72`|
| Git 分支 | `origin/main` |

---

## 2. 假设与预期

> 在跑实验**之前**写下你的预期。之后对比结果，培养科学直觉。

**假设**：
- 模型能够在 50 epoch 内学会 MNIST 手写数字的简单数据分布

**预期结果**：
- 生成的数字较为清晰，且有EMA的比没有的FID分数更低，采样效果更好
- fp16可以提升训练速度

**如果与预期不符可能的原因**：
- 训练轮次不足，EMA 参数还是没有调好
- 不直接预测噪声，而是先计算x0，限制在[-1,1]，避免溢出

---

## 3. 配置（完整 hyperparameters）

```yaml
output_dir: runs/exp_mnist_baseline_2
seed: 42

# ─────────── 数据 ───────────
dataset:
  name: mnist
  batch_size: 128
  image_size: 32          # 28x28 padding 到 32x32 便于 stride-2 下采样
  num_workers: 4
  root: ./data

# ─────────── 模型（小型 U-Net） ───────────
model:
  in_channels: 1
  out_channels: 1
  base_channels: 64
  channel_mult: [1, 2, 2]    # 32 → 16 → 8 → 8
  num_res_blocks: 2
  attn_resolutions: [16]      # 在 16x16 分辨率加 self-attention
  dropout: 0.1
  image_size: 32

# ─────────── Diffusion ───────────
diffusion:
  T: 1000
  beta_schedule: linear
  beta_start: 0.0001
  beta_end: 0.02

# ─────────── Optimizer ───────────
optimizer:
  lr: 2.0e-4
  weight_decay: 0.0
  warmup_steps: 1000

# ─────────── EMA ───────────
ema_decay: 0.999

# ─────────── 混合精度（可选） ───────────
mixed_precision: fp16       # 选项: 'no', 'fp16', 'bf16'
                          # 第一次跑用 'no'，确认正常后再开

# ─────────── 训练循环 ───────────
training:
  num_epochs: 50
  log_every: 50             # 每 N 步打印一次 loss
  sample_every: 1000        # 每 N 步采样并保存可视化
  ckpt_every: 5000          # 每 N 步保存 checkpoint
  grad_clip: 1.0

# ─────────── WandB（可选） ───────────
wandb:
  enabled: true            # 想用 wandb 改成 true
  project: ddpm-course
  run_name: mnist_baseline_2

# ─────────── 绘图（可选） ───────────
plotting:
  enabled: true             # 是否启用 matplotlib 本地绘图
  record_every: 50          # 每 N 步记录一次数据（与 log_every 一致）
  plot_every: 1000          # 每 N 步更新并保存一次曲线图（与 sample_every 一致）
```

---

## 4. 运行命令（必须可直接复制粘贴运行）

```bash
python train.py --config configs/mnist2.yaml
```

---

## 5. 结果

### 5.1 数值指标

| 指标 | 数值 | 备注 |
|------|------|------|
| 训练总时长 | 98.6 min |3.95 steps/s |
| 显存峰值 | 3.54 GB | |
| 最终训练 loss | 0.0145 | |
| FID @ EMA| 2.8824 | 5000 张样本 |
| FID @ no EMA| 5.3107 | 5000 张样本 |

### 5.2 可视化结果

`wandb/run-20260525_190443-v11lkmr3`

[https://wandb.ai/worldexplorer111111-tsinghua-university/ddpm-course/runs/v11lkmr3](https://wandb.ai/worldexplorer111111-tsinghua-university/ddpm-course/runs/v11lkmr3)

（注册时以为能改，选了清华，后面发现改不了了...）

### 5.3 与 baseline 对比

| 配置 | FID(ema/no_ema) | 训练时长 | 备注 |
|------|-----|----------|------|
| ema_decay: 0.9999 (baseline) | 52.8770/5.0223 | 164 min | exp_20260524_01 |
| ema_decay: 0.999 (本次) | 2.8824/5.3107 | 98.6 min |exp_20260525_01 |

---

## 6. 结论

### 6.1 假设是否成立

✅ 假设成立

**简短说明**：
1. 手写数字的分布非常清晰，第一次训练到后期才出现手写数字，而本次在前期就显现，且 FID 分数从 52.877 降到 2.8824
2. 使用fp16后，训练时间从 164 min 提高到 98.6 min ，提高约 40 %，且显存峰值降低
3. 使用 EMA 比不使用 EMA 的 FID 提高 2.4283，符合预期

### 6.2 现象记录

观察到但与预期无关的现象：

- 

---

## 7. 失败记录与踩坑（**强制填写**）

> 没有遇到任何问题是不正常的。仔细回想是否有任何"花了 5 分钟以上才解决的事情"。

---

## 8. 下一步计划

> 基于本次结果，下一个实验应该做什么？写得越具体越好。

- [ ] 对比linear/cosine 两种 schedule

---

## 9. 给未来自己的提醒

> 一句话总结这次实验最值得记住的事情。

*看清原理*

---
