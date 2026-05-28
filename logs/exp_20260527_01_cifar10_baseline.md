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
- CIFAR-10 包含复杂的颜色和背景特征，相比 MNIST，模型需要更长的时间和更大的通道数来学习数据分布。
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

# ─────────── 绘图（可选） ───────────
plotting:
  enabled: true             
  record_every: 50          
  plot_every: 1000 
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
| 训练总时长 | 14 h 53 min |使用140W便携适配器供电 |
| 显存峰值 | 4.21 GB | |
| 最终训练 loss | 0.02853 | |
| FID @ EMA| 17.8579 | 5000 张样本 |
| FID @ no EMA| 33.0284 | 5000 张样本 |

### 5.2 可视化结果

`wandb/run-20260526_230418-qsvhdus3`
`wandb/run-20260527_130451-qsvhdus3`
`wandb/run-20260527_230221-qsvhdus3`

[wandb.ai/worldexplorer111111-tsinghua-university/ddpm-course/runs/qsvhdus3](https://wandb.ai/worldexplorer111111-tsinghua-university/ddpm-course/runs/qsvhdus3)

（注册时以为能改，选了清华，后面发现改不了了...）

### 5.3 与 baseline 对比

| 配置 | FID(ema/no_ema) | 训练时长 | 备注 |
|------|-----|----------|------|
| linear schedule/ema_decay: 0.9999 (baseline) | 17.8579/33.0284 | 14 h 53 min |exp_20260527_01，使用140W便携适配器供电 |

---

## 6. 结论

### 6.1 假设是否成立

🟡 部分成立

**简短说明**：
1. 降低 batch size 到 64 并开启 fp16 后，显存峰值仅为 4.21 GB，不仅扛过了 6GB 显存的限制，甚至还有余量。整体耗时约 14.8 小时，在预估的 15-20 小时区间内。
2. EMA 模型的 FID 为 17.8579，虽然十分接近我们设定的优秀阈值（≤ 15），但仍有些微差距。相比于无 EMA 模型的 33.0284，EMA 依然发挥了巨大的平滑作用，证明模型确实学到了 CIFAR-10 的分布结构，但对于复杂的三通道图像，当前的 Linear Schedule 或训练时长可能还没能完全逼出它的上限。
> EMA 模型的 64 个采样中，大部分图片都能较好表现，但是出现一个全蓝图。(DDPM 的采样完全取决于初始的那一帧随机高斯噪声（$x_T$）。如果在 64 张采样图中，某一个初始噪声张量恰好落在了模型学习到的分布的极其边缘的地带，模型在进行去噪推断时就会不知到往哪个方向走，遇到这种不会解的噪声时，神经网络通常会选择一种最安全的策略——输出数据集的主导底色。CIFAR-10 的 10 个类别中，飞机（背景多为蓝天）、船只（背景多为蓝海）、鸟类（背景多为天空）占据了很大比例，使得大面积蓝色成为了模型试图降低全局 Loss 时的强烈先验。退化成全蓝图，就是模型迷失方向时交出的一张白卷)

> 无 EMA 模型会出现鸟的轮廓和狗的面部组合，飞机浮在海上的情况，还有一些无法分辨的图片。(batch size 缩小到 64 使得每次参数更新的梯度包含了更多的噪声，在早期的去噪步骤中，模型可能勾勒出了一个“有翅膀”的低频轮廓和“蓝色背景”；但在中间或后期的去噪步骤中，由于非 EMA 模型权重的剧烈波动，它在填补局部高频细节时，突然转向了当前权重更敏感的“狗脸”。这就导致了特征图的错误拼接。而 EMA decay 设置为 0.9999 意味着当前 EMA 模型的权重融合了过去大约 10000 步的经验，抹平了 batch size 缩小带来的梯度抖动，相当于在复杂的损失函数地形中走出了一条最稳健的平滑轨迹)

### 6.2 现象记录

观察到但与预期无关的现象：

- 恢复断点后，wandb 会忽略已有步的上传，直到出现记录未出现的步，这些未上传的步可以在本地查看手动保存的数据。

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

### 坑 2：变量未初始化
```bash
UnboundLocalError: local variable 'saved_ckpts' referenced before assignment
```
- **现象**：断点恢复后，模型正常跑完到 70000 步。但触发 torch.save() 并试图执行清理旧 Checkpoint 的逻辑时，程序突然崩溃，提示 saved_ckpts 变量未定义。
- **猜测原因**：把初始化放在不会执行的函数或判断语句的末尾
- **验证过程**：仔细核对代码作用域和配置文件后发现，是因为在之前的修改中，不小心将 saved_ckpts 的初始化逻辑与 plotting（绘图功能）的条件判断绑定在了一起。而当前训练使用的是 cifar10.yaml，我并没有把 mnist 的训练配置中关于 plotting: enabled: true 的这段 YAML 代码复制过来。由于配置中缺少该字段，程序默认跳过了绘图相关的初始化代码块，导致 saved_ckpts 根本没有被创建，但在后方的保存逻辑中却又直接调用了它。
- **最终方法**：在当前的训练配置文件中，补齐缺失的 plotting 配置；将控制变量 if 语句块中剥离出来，放置在训练大循环正式开始前的最外层，确保其无条件绝对执行。
- **耗时**：约 20 分钟
- **教训**：维护多个并行的 YAML 配置文件时，改动代码结构要对齐同步所有的相关配置文件。

### 坑 3：绘图断点恢复逻辑问题
- **现象**：执行 --resume 重启后，模型跑到 61,000 步触发 plot_every 时，新生成的 training_curves.png 会覆盖掉旧图，失去 0~60000 步的本地曲线记录，图片上只会显示 60,000 步之后的数据。
- **猜测原因**：由于 history_steps、history_losses 等列表在重启训练后被重新初始化为空，没有进行保存恢复。
- **最终方法**：将绘图的历史数据一并打包进 Checkpoint 中。
- **耗时**：约 15 分钟
- **教训**：上一次训练完全没有发现这个问题，修改代码后运行注意检查所有输出。

---

## 8. 下一步计划

> 基于本次结果，下一个实验应该做什么？写得越具体越好。

- [ ] 挑战 CIFAR-10 的 Cosine Schedule 实验，验证更平滑的加噪策略是否能将 FID 分数进一步压低至 15 以内


---

## 9. 给未来自己的提醒

> 一句话总结这次实验最值得记住的事情。

*重视显存和硬盘空间的调度*

---
