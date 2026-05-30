# Experiment Log 

`exp_20260528_01_cifar10_cosine`

## 一句话目标

> 本次实验最想验证什么？答案应当是 yes/no 形式可判定的。

*在验证在 CIFAR-10 数据集上将加噪策略从 Linear Schedule 替换为 Cosine Schedule，能否进一步提升生成质量，将 EMA 模型的 FID 分数成功压低至 15 以内。*

---

## 1. 环境信息

| 项目 | 内容 |
|------|------|
| 实验日期 | 2026-05-28 |
| 实验者 | HZQ |
| 硬件 | NVIDIA GeForce RTX 3060 6GB × 1 |
| 框架版本 | PyTorch 2.3.0, CUDA 12.1 |
| 仓库 | github.com/coding-all-the-time/diffusion-model-prectical-project1 |
| Git commit | `9388b62` |
| Git 分支 | `origin/main` |

---

## 2. 假设与预期

> 在跑实验**之前**写下你的预期。之后对比结果，培养科学直觉。

**假设**：
- Cosine schedule 在加噪的中间阶段更加平滑，使得网络能在更长的时间步里保留并学习到 CIFAR-10 复杂的低频轮廓和高频颜色细节，而不是像 Linear 那样过早地破坏图像信息。
- 模型在 200 epoch 内能够适应 Cosine 的噪声分布，并且 EMA 能继续发挥抹平 batch size (64) 带来的梯度抖动的作用。

**预期结果**：
- 显存占用贴近最大值 6GB显存占用依然在 4.2 GB 左右，没有显著增加 。
- 训练总时长与上次实验持平，预计在 14.5 ~ 15.5 小时之间。
- 生成的图像在颜色和结构上比 Linear baseline 更合理（比如不再出现太多奇怪的拼接），EMA 模型的 FID 分数有望突破 15 的大关（≤ 15）。

**如果与预期不符可能的原因**：
- Cosine Schedule 的最优学习率或训练步数可能与 Linear 不同。恒定的 2.0e-4 学习率在 Cosine 分布下可能不够完美。
- 200 epochs 对于 CIFAR-10 来说，可能依然只是"刚及格"的轮次，要显著提高 FID 可能需要更多 epochs。

---

## 3. 配置（完整 hyperparameters）

```yaml
# CIFAR-10 DDPM Configuration（Cosine 进阶档）
# 期望训练时长: ~15h on RTX 3060 6GB
# 期望结果: 200 epoch 后 FID ≤ 15 (5000 samples)

output_dir: runs/exp_cifar10_cosine
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
mixed_precision: fp16        # 保持开启 fp16 防止 OOM

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
  run_name: cifar10_cosine

# ─────────── 绘图（可选） ───────────
plotting:
  enabled: true             
  record_every: 50          
  plot_every: 1000
```

---

## 4. 运行命令（必须可直接复制粘贴运行）

```bash
python train.py --config configs/cifar10_cosine.yaml
```
如果中途中断，使用以下命令恢复
```bash
python train.py --config configs/cifar10_cosine.yaml --resume runs/exp_cifar10_cosine/ckpt/step_XXXXXX.pt # 替换为具体 checkpoint名
```

---

## 5. 结果

### 5.1 数值指标

| 指标 | 数值 | 备注 |
|------|------|------|
| 训练总时长 | 15 h 11 min |使用140W便携适配器供电 |
| 显存峰值 | 3.70 GB | |
| 最终训练 loss | 0.06495 | |
| FID @ EMA| 62.0114 | 5000 张样本 |
| FID @ no EMA| 189.3443 | 5000 张样本 |

### 5.2 可视化结果

`wandb/run-20260528_175523-079h0hh4`

[wandb.ai/worldexplorer111111-tsinghua-university/ddpm-course/runs/079h0hh4](https://wandb.ai/worldexplorer111111-tsinghua-university/ddpm-course/runs/079h0hh4)

（注册时以为能改，选了清华，后面发现改不了了...）

### 5.3 与 baseline 对比

| 配置 | FID(ema/no_ema) | 训练时长 | 备注 |
|------|-----|----------|------|
| linear schedule/ema_decay: 0.9999 (baseline) | 17.8579/33.0284 | 14 h 53 min |exp_20260527_01，使用140W便携适配器供电 |
| cosine schedule/ema_decay: 0.9999 (本次) | 62.0114/189.3443 | 15 h 11 min |exp_20260528_01，使用140W便携适配器供电 |

---

## 6. 结论

### 6.1 假设是否成立

❌ 不成立

**简短说明**：
1. 期望 Cosine 带来更好的平滑性和细节学习，结果发生采样坍塌。FID 分数不降反升，EMA 模型升至 62，非 EMA 达到 189。
2. 虽然训练过程没有报错，但模型内部发生严重的数值溢出问题，导致出图包含大量纯色废图。
3. 观察前 42500 步的采样结果，图像呈现非常模糊的彩色马赛克。从 45000 步开始，网格中突兀地出现了纯白、纯黑、纯黄、纯青等没有任何纹理的色块。在训练后期（10万步以后），同一个 batch 采样出的图像，有的展现出了清晰的动物和汽车，有的却依然是死板的纯色方块。
4. 采样后发现不加 EMA 的模型比加 EMA 的模型出现更多纯色色块。

### 6.2 现象记录

观察到但与预期无关的现象：

- 

---

## 7. 失败记录与踩坑（**强制填写**）

> 没有遇到任何问题是不正常的。仔细回想是否有任何"花了 5 分钟以上才解决的事情"。

### 坑 1：纯色块问题

- **现象**：训练中后期产生大量纯色块，尤其是 Non-EMA 模型的输出完全崩溃。
- **猜测原因**：fp16 动态范围过窄，Cosine 调度的特性使其在 $t \to T$ 的高噪声域停留的时间比 Linear 更长。在面对近乎纯噪声的输入时，U-Net 内部的激活值（特别是 Self-Attention 的 $Q \cdot K^T$）极易飙升击穿 65504 的上限，导致前向或反向传播中突然产生 NaN/Inf。由于无 EMA 模型直接承受这些梯度爆炸的冲击，其权重发生剧烈震荡，导致采样全是色块；而 EMA 模型作为缓冲了权重的崩坏速度，所以表现出好一些，但整体依然差的特征。
- **验证过程**：在 dataset.py 中确认数据集无 NaN/Inf 数据干扰，纯色数据多样，并不是单一颜色。
- **最终方法**：由于 RTX 3060 采用 Ampere 架构，原生支持拥有与 fp32 同样动态范围的 bf16。将 fp16 更改为 bf16 。
- **耗时**：约 30 分钟
- **教训**：注意数值精度

---

## 8. 下一步计划

> 基于本次结果，下一个实验应该做什么？写得越具体越好。

- [ ] 开启新一轮实验，验证切换为 bf16 后是否能消除网络内的数值溢出，消除纯色块，发掘 Cosine Schedule 的潜力。


---

## 9. 给未来自己的提醒

> 一句话总结这次实验最值得记住的事情。

*在变量数值精度和计算资源中平衡*

---
