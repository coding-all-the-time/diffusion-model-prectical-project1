# Experiment Log 
## Experiment 1

`exp_20260526_01_mnist_cosine`

## 一句话目标

> 本次实验最想验证什么？答案应当是 yes/no 形式可判定的。

*在保留 fp16 和 EMA (0.999) 优化的基础上，验证使用 Cosine Schedule 替代 Linear Schedule 是否能在 MNIST 数据集上进一步提升生成质量，降低 FID 分数。*

---

## 1. 环境信息

| 项目 | 内容 |
|------|------|
| 实验日期 | 2026-05-26 |
| 实验者 | HZQ |
| 硬件 | NVIDIA GeForce RTX 3060 6GB × 1 |
| 框架版本 | PyTorch 2.3.0, CUDA 12.1 |
| 仓库 | github.com/coding-all-the-time/diffusion-model-prectical-project1 |
| Git commit | `17f6ba4`|
| Git 分支 | `origin/main` |

---

## 2. 假设与预期

> 在跑实验**之前**写下你的预期。之后对比结果，培养科学直觉。

**假设**：
- Cosine schedule 相比 Linear schedule，在加噪过程的中间阶段变化更加平滑，不会过早破坏图像信息
- 模型在 50 epoch 内能够较好地适应 Cosine schedule 的噪声分布

**预期结果**：
- 训练时长和显存占用应与第二次实验基本持平（约 98 分钟，3.5GB）
- 生成的数字质量可能更好，预期 EMA 的 FID 分数能持平或低于第二次实验的 2.8824

**如果与预期不符可能的原因**：
- MNIST 数据集过于简单，Linear schedule 已经足够拟合，Cosine 带来的额外平滑收益不明显，甚至可能导致学习节奏不匹配
- 学习率 2.0e-4 对 Cosine schedule 可能需要重新微调

---

## 3. 配置（完整 hyperparameters）

```yaml
output_dir: runs/exp_mnist_cosine
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
  beta_schedule: cosine
  # beta_start: 0.0001
  # beta_end: 0.02
  s: 0.008

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
  run_name: mnist_cosine

# ─────────── 绘图（可选） ───────────
plotting:
  enabled: true             # 是否启用 matplotlib 本地绘图
  record_every: 50          # 每 N 步记录一次数据（与 log_every 一致）
  plot_every: 1000          # 每 N 步更新并保存一次曲线图（与 sample_every 一致）
```

---

## 4. 运行命令（必须可直接复制粘贴运行）

```bash
python train.py --config configs/mnist_cosine.yaml
```

---

## 5. 结果

### 5.1 数值指标

| 指标 | 数值 | 备注 |
|------|------|------|
| 训练总时长 | 193.9 min |2.01183 steps/s, 训练中断 |
| 显存峰值 | 4.04 GB | |
| 最终训练 loss | 0.02514 | |
| FID @ EMA| 2.9506 | 5000 张样本 |
| FID @ no EMA| 7.4592 | 5000 张样本 |

### 5.2 可视化结果

`wandb/run-20260526_105148-t5x7nibf`

[wandb.ai/worldexplorer111111-tsinghua-university/ddpm-course/runs/t5x7nibf](https://wandb.ai/worldexplorer111111-tsinghua-university/ddpm-course/runs/t5x7nibf)

（注册时以为能改，选了清华，后面发现改不了了...）

### 5.3 与 baseline 对比

| 配置 | FID(ema/no_ema) | 训练时长 | 备注 |
|------|-----|----------|------|
| linear schedule/ema_decay: 0.9999 (baseline) | 52.8770/5.0223 | 164 min | exp_20260524_01 |
| linear schedule/ema_decay: 0.999 (baseline_2) | 2.8824/5.3107 | 98.6 min |exp_20260525_01 |
| cosine schedule/ema_decay: 0.999 (本次) | 2.9506/7.4592 | 193.9 min(训练中断) |exp_20260526_01 |
---

## 6. 结论

### 6.1 假设是否成立

🟡 部分成立

**简短说明**：
1. Cosine schedule 带 EMA 的 FID (2.9506) 与上一版 Linear (2.8824) 基本持平，肉眼观察采样结果也相似，说明在 MNIST 这种简单数据集上，两种 schedule 的上限都很高。
2. 不带 EMA 的采样结果中，63 张里出现了一张笔画不清晰和一张纯白色样本，且 FID 升至 7.4592（差于 Linear 的 5.3107）。这表明模型对 Cosine 调度的噪声分布适应得不如 Linear 稳定，可能目前的恒定学习率（2.0e-4）对 Cosine 来说并非最优。
3. 由于遭遇断电导致硬件降频（降至 2.01 steps/s），本次训练耗时达 193.9 分钟，未能与上一版（98.6 分钟）形成公平对比，但显存峰值（4.04 GB）保持在合理范围内。

### 6.2 现象记录

观察到但与预期无关的现象：

- 警告：
```bash
FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
```
PyTorch 的自动混合精度（AMP，用于加速模型训练并节省显存）主要是专门为 NVIDA 显卡（CUDA）设计的。随着 PyTorch 的发展，AMP 开始支持更多的硬件设备（比如 CPU，或者苹果的 MPS）。为了统一接口，PyTorch 官方把这个功能移到了更通用的 torch.amp 模块下。在使用时需要明确告诉它你要加速的是哪种设备（比如传入 'cuda' 或 'cpu'）。
修改训练代码为：
```python
with torch.amp.autocast(enabled=use_amp, dtype=amp_dtype, device_type='cuda'):

scaler = torch.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16), device='cuda')
```

---

## 7. 失败记录与踩坑（**强制填写**）

> 没有遇到任何问题是不正常的。仔细回想是否有任何"花了 5 分钟以上才解决的事情"。

### 坑 1：参数错误
```bash
TypeError: cosine_beta_schedule() got an unexpected keyword argument 'beta_start'
```
- **现象**：运行训练脚本后，如上报错
- **猜测原因**：Linear schedule 依赖 start 和 end 值进行线性插值，但 Cosine schedule 的数学定义仅依赖总步数 $T$ 和偏移量 $s$。代码在构建 diffusion 过程时，没有针对 cosine 做参数隔离，把 config 里的 `beta_start` 等参数强制传给了不支持这些参数的函数。
- **验证过程**：在配置文件中将 beta_start 和 beta_end 改为 s，问题消失
- **最终方法**：修改配置文件
- **耗时**：约 5 分钟
- **教训**：修改配置时，必须要检查配置文件中的参数是否变化

### 坑 2：训练意外中断

- **现象**：在教室跑实验，电源被阿姨拔掉，电脑没电关机导致训练在第 16600 步中断，插电唤醒后，系统休眠机制保住了当前进程，让训练以 1.8 steps/s 的降频速度苟延残喘(随着电量提高，速度也慢慢提高，应该是电脑自动进入省电模式)
- **猜测原因**：检查代码后，发现虽然在配置 parse_args() 中定义了 --resume 接收路径字符串，但 train() 函数里没有写加载权重字典 .pt 文件的闭环逻辑，导致该参数没有作用
- **验证过程**：查阅 train() 源码约第 151 行处，发现变量定义为硬编码的 global_step = 0，并且循环固定为 for epoch in range(num_epochs):，完全没有针对断点状态的反向解析和变量覆盖
- **最终方法**：使这次训练先跑完，为了彻底解决未来可能的降频问题和安全隐患，将恢复逻辑补全到代码中并在下一次训练进行验证。具体修改如下：
1. 在 main() 函数底部，将命令行的 resume 路径写进配置：
```python
if args.resume is not None:
    cfg['resume'] = args.resume
```
2. 在 train() 函数中替换掉原本写死的起始变量，并加载各个组件的状态字典：
```python
    global_step = 0
    start_epoch = 0

    # ─────────── 修复：新增完整的断点恢复逻辑 ───────────
    if cfg.get('resume'):
        print(f"[*] Resuming from checkpoint: {cfg['resume']}")
        ckpt = torch.load(cfg['resume'], map_location=device, weights_only=False)

        # 恢复模型与优化器
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])

        # 恢复 EMA 和 混合精度 Scaler
        if ema is not None and 'ema' in ckpt:
            ema.load_state_dict(ckpt['ema'])
        if use_amp and amp_dtype == torch.float16 and 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])

        # 恢复训练步数状态
        global_step = ckpt.get('global_step', 0)
        start_epoch = ckpt.get('epoch', 0) + 1  

        # 防止学习率调度器内部 step 归零导致重新 warmup
        if scheduler_lr is not None:
            scheduler_lr.last_epoch = global_step

    start_time = time.time()
    print(f"[train] Starting training from epoch {start_epoch} to {num_epochs}")

    # 修复：将固定的 0 替换为 start_epoch
    for epoch in range(start_epoch, num_epochs):
```
- **耗时**：约 30 分钟
- **教训**：跑长耗时的实验前，不能想当然地以为代码里留了接口就等于实现了功能。务必 review 一遍模型权重、优化器和各种调度器的 Checkpoint 保存与加载闭环逻辑是否严密，防患于未然

---

## 8. 下一步计划

> 基于本次结果，下一个实验应该做什么？写得越具体越好。

- [ ] 使用本次实验新编写的 checkpoint resume 逻辑，人为中断并恢复一次短训练，确保模型权重、优化器、EMA 和混合精度 Scaler 均能无缝衔接。

---

## 9. 给未来自己的提醒

> 一句话总结这次实验最值得记住的事情。

*以后训练记得加断点恢复*

---
