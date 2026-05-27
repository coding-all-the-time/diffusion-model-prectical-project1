# Experiment Log 

`exp_20260527_01_cifar10_baseline`

## 一句话目标

> 本次实验最想验证什么？答案应当是 yes/no 形式可判定的。

*在 RTX 3060 (6GB) 硬件受限的情况下，验证降低 batch size 至 64 并开启 fp16 后，能否成功跑通 CIFAR-10 的 200 个 epoch 训练，并观察到较为清晰的生成图像（FID ≤ 15）。*

---

## 1. 环境信息

| 项目 | 内容 |
|------|------|
| 实验日期 | 2026-05-27 |
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
- CIFAR-10 包含复杂的颜色和背景特征，相比 MNIST，模型需要更长的时间（200 epochs）和更大的基础通道数（128）来学习数据分布。
- Batch size 从 128 减半到 64 可能会导致每次更新的梯度方向稍微嘈杂，但配合 AdamW 优化器和 EMA (0.9999) 的平滑作用，整体依然能够稳定收敛。

**预期结果**：
- 显存占用贴近最大值 6GB 。
- 训练时长将大幅增加。参考配置文件中 RTX 3090 需要约 8 小时，RTX 3060 预计耗时可能在 15-20 小时左右。
- 最终生成的图像能明显辨认出汽车、动物等 CIFAR-10 类别轮廓，不再是纯噪声。

**如果与预期不符可能的原因**：
- 尽管 batch size 减半并开启了 fp16，128 基础通道数的 U-Net 在计算特征图时仍可能超出 6GB 显存。
- fp16 混合精度在 CIFAR-10 训练初期（特别是梯度较大时）可能导致数值溢出，出现 Loss 为 NaN 的情况。
- 200 个 epoch 对于 3060 来说很久，但对于完整的 DDPM 训练可能刚刚达到产生尚可结果的门槛。

---

## 3. 配置（完整 hyperparameters）

```yaml
# CIFAR-10 DDPM Configuration（进阶档）
output_dir: runs/exp_cifar10_baseline
seed: 42

# ─────────── 数据 ───────────
dataset:
  name: cifar10
  batch_size: 64 # 原来是128， 由于我的3060是6GB显存，所以只能减半 
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
  beta_schedule: linear       # 挑战档可改为 cosine
  beta_start: 0.0001
  beta_end: 0.02

# ─────────── Optimizer ───────────
optimizer:
  lr: 2.0e-4
  weight_decay: 0.0
  warmup_steps: 5000

# ─────────── EMA ───────────
ema_decay: 0.9999

# ─────────── 混合精度 ───────────
mixed_precision: fp16        # CIFAR-10 训练建议开 fp16

# ─────────── 训练循环 ───────────
training:
  num_epochs: 200
  log_every: 100
  sample_every: 2500
  ckpt_every: 10000
  grad_clip: 1.0
  max_keep_ckpts: 3           

# ─────────── WandB（强烈推荐） ───────────
wandb:
  enabled: true             
  project: ddpm-course
  run_name: cifar10_linear_baseline
```

---

## 4. 运行命令（必须可直接复制粘贴运行）

```bash
python train.py --config configs/cifar10.yaml
```
如果中途中断，使用以下命令恢复
```bash
python train.py --config configs/cifar10.yaml --resume runs/exp_cifar10_baseline/ckpt/step_XXXXXX.pt # 替换为具体 checkpoint名
```

---

## 5. 结果

### 5.1 数值指标

| 指标 | 数值 | 备注 |
|------|------|------|
| 训练总时长 | 114.6 min |使用140W便携适配器供电 |
| 显存峰值 | 4.12 GB | |
| 最终训练 loss | 0.02478 | |
| FID @ EMA| 待填 | 5000 张样本 |
| FID @ no EMA| 待填 | 5000 张样本 |

### 5.2 可视化结果

`wandb/run-20260526_200240-ibzzk3t7`
`wandb/run-20260526_202733-ibzzk3t7`

[wandb.ai/worldexplorer111111-tsinghua-university/ddpm-course/runs/ibzzk3t7](https://wandb.ai/worldexplorer111111-tsinghua-university/ddpm-course/runs/ibzzk3t7)

（注册时以为能改，选了清华，后面发现改不了了...）

### 5.3 与 baseline 对比

| 配置 | FID(ema/no_ema) | 训练时长 | 备注 |
|------|-----|----------|------|
| linear schedule/ema_decay: 0.999 (baseline_2) | 2.8824/5.3107 | 98.6 min |exp_20260525_01，使用电脑原装大型适配器供电 |
| cosine schedule/ema_decay: 0.999 (mnist_cosine) | 2.9506/7.4592 | 193.9 min |exp_20260526_01，使用140W便携适配器供电, 训练中断 |
| cosine schedule/ema_decay: 0.999 (本次) | 待填 | 114.6 min |exp_20260526_02，使用140W便携适配器供电 |
---

## 6. 结论

### 6.1 假设是否成立

🟡 部分成立

**简短说明**：
1. 

### 6.2 现象记录

观察到但与预期无关的现象：

- 

---

## 7. 失败记录与踩坑（**强制填写**）

> 没有遇到任何问题是不正常的。仔细回想是否有任何"花了 5 分钟以上才解决的事情"。

### 坑 1：训练模型保存时崩溃
```bash
RuntimeError: [enforce fail at inline_container.cc:783] . PytorchStreamWriter failed writing file data/140: file write failed
RuntimeError: [enforce fail at inline_container.cc:603] . unexpected pos 155797120 vs 155797008
```
- **现象**：模型已经顺利训练了 70,000 步，前向传播、Loss 下降和采样出图都完全正常，但在触发 torch.save() 保存当前 step 的 Checkpoint 时，突然抛出报错并导致训练进程崩溃终止。
- **猜测原因**：最初看到 inline_container.cc 和 unexpected pos 的报错，以为是 PyTorch 的序列化模块存在 Bug，或者因为开了混合精度和 EMA，导致要保存的字典中包含损坏的 Tensor 数据。
- **验证过程**：排查 Traceback，发现最关键的核心报错是 file write failed，是操作系统的 I/O 拒绝。查看 D 盘后，发现原有的几十 GB 剩余空间已被耗尽，PyTorch 将权重打包写入硬盘时中途失败，最终引发文件内部指针位置错乱（unexpected pos）。
- **最终方法**：在 train.py 的保存逻辑中增加滚动覆盖机制：在训练启动时，利用 `saved_ckpts = sorted(list(ckpt_dir.glob('step_*.pt')))` 扫描并升序排列已有的文件；在每次保存新 Checkpoint 后，利用 `if len(saved_ckpts) > max_keep_ckpts:` 判断，通过 `.pop(0)` 和 `.unlink()` 自动删掉最老的旧文件，确保硬盘里最多只保留 3 个 Checkpoint(默认配置，可以修改)。
- **耗时**：约 30 分钟
- **教训**：跑较大的模型训练时，不仅要紧盯显存会不会 OOM，还要高度警惕硬盘空间的消耗。在项目初期就引入 Checkpoint 的垃圾回收与自动清理机制。

---

## 8. 下一步计划

> 基于本次结果，下一个实验应该做什么？写得越具体越好。

- [ ] 挑战一下 cifar10 的训练。

---

## 9. 给未来自己的提醒

> 一句话总结这次实验最值得记住的事情。

**

---
