# Experiment Log 
## Experiment 1

`exp_20260524_01_mnist_baseline`

## 一句话目标

> 本次实验最想验证什么？答案应当是 yes/no 形式可判定的。

*跑通 DDPM 的完整训练代码，并在 MNIST 数据集上建立一个 linear schedule 的 baseline，观察到手写数字。未开启fp16或bf16，记录当前代码在 RTX 3060 硬件上的训练耗时（约 2.5 小时）。*

---

## 1. 环境信息

| 项目 | 内容 |
|------|------|
| 实验日期 | 2026-05-24 |
| 实验者 | HZQ |
| 硬件 | NVIDIA GeForce RTX 3060 6GB × 1 |
| 框架版本 | PyTorch 2.3.0, CUDA 12.1 |
| 仓库 | github.com/coding-all-the-time/diffusion-model-prectical-project1 |
| Git commit | `5d0d534`|
| Git 分支 | `origin/main` |

---

## 2. 假设与预期

> 在跑实验**之前**写下你的预期。之后对比结果，培养科学直觉。

**假设**：
- UNet 能够在 50 epoch 内学会 MNIST 手写数字的简单数据分布。

**预期结果**：
- 最终生成的 32x32 图像样本网格中，能清晰辨认出 0-9 的数字轮廓，无大面积纯噪声
- 最终 loss 收敛到一个较稳定的低值

**如果与预期不符可能的原因**：
- 学习率 0.0002 与 batch size 128 未调优
- MNIST 图片特征过于简单，导致模型过早过拟合

---

## 3. 配置（完整 hyperparameters）

```yaml
output_dir: runs/exp_mnist_baseline
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
ema_decay: 0.9999

# ─────────── 混合精度（可选） ───────────
mixed_precision: no       # 选项: 'no', 'fp16', 'bf16'
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
  enabled: false            # 想用 wandb 改成 true
  project: ddpm-course
  run_name: mnist_baseline
```

---

## 4. 运行命令（必须可直接复制粘贴运行）

```bash
python train.py --config configs/mnist.yaml
```

---

## 5. 结果

### 5.1 数值指标

| 指标 | 数值 | 备注 |
|------|------|------|
| 训练总时长 | 164min |2.4steps/s |
| 显存峰值 | 4.34 GB | |
| 最终训练 loss | 0.0145 | |
| FID @ EMA| 52.8770 | 5000 张样本 |
| FID @ no EMA| 5.0223 | 5000 张样本 |

### 5.2 可视化结果

> 暂时没有加入绘图，下一次实验将加入。

### 5.3 与 baseline 对比

> 暂时没有可对比的baseline。

---

## 6. 结论

### 6.1 假设是否成立

🟡 部分成立

**简短说明**：
1. 出现手写数字的分布，训练时间符合预期，loss较低。
2. 加 EMA 的结果分布很不均匀（1、3、4、7较多），且FID分数是52.8770。而不加 EMA 的采样结果所有数字都出现，且FID只为5.0223。原因可能是 0.9999 太小。在 2w 多步的早期训练中，EMA 模型可能包含了太多极其早期的（瞎猜的）权重，导致评估时表现反而不如原始的 Model。

### 6.2 现象记录

观察到但与预期无关的现象：

- 

---

## 7. 失败记录与踩坑（**强制填写**）

> 没有遇到任何问题是不正常的。仔细回想是否有任何"花了 5 分钟以上才解决的事情"。

### 坑 1：模块相对导入报错

- **现象**：ModuleNotFoundError: No module named 'embedding'
- **猜测原因**：在 model/unet.py 中直接使用 from embedding import TimeEmbedding，导致 Python 在项目根目录寻找模块失败。
- **最终方法**：修改为相对导入 from .embedding import TimeEmbedding
- **耗时**：约 2 分钟
- **教训**：文件树复杂时，要注意import是否生效

### 坑 2：YAML 配置文件读取遭遇 GBK 解码错误

- **现象**：UnicodeDecodeError: 'gbk' codec can't decode byte 0x80 in position 34
- **猜测原因**：Windows 系统下的 open() 函数默认使用 GBK 编码，而配置文件是以 UTF-8 保存的（包含中文字符）
- **最终方法**：在 train.py 中的文件读写处显式添加 encoding='utf-8'
- **耗时**：约 2 分钟

### 坑 3：PyTorch 无法识别 CUDA，模型在 CPU 上龟速运行

- **现象**：终端警告 CUDA is not available. Disabling，且 torch.cuda.is_available() 返回 False。
- **猜测原因**：正确激活包含 GPU 环境的 Conda 虚拟环境
- **最终方法**：卸载当前 torch，查明显卡支持 CUDA 12.3 后，安装适配的 PyTorch 版本
- **耗时**：约 1 小时
- **教训**：环境很重要，学会做环境管理

### 坑 4：Windows PowerShell 拦截 Conda 脚本执行

- **现象**：提示 无法加载文件 profile.ps1，因为在此系统上禁止运行脚本。
- **猜测原因**：PowerShell 的默认执行策略（Execution Policy）为 Restricted，拦截了 conda 的初始化脚本，导致环境无法激活（终端无 (base) 前缀）。
- **最终方法**：执行 Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser 开放权限并重启终端解决。
- **耗时**：约 10 分钟

---

## 8. 下一步计划

> 基于本次结果，下一个实验应该做什么？写得越具体越好。

- [ ] 将 EMA 衰减率改为 0.999
- [ ] 打开fp16，观察训练速度和显存情况
- [ ] 开启wandb
- [ ] 加入绘图

---

## 9. 给未来自己的提醒

> 一句话总结这次实验最值得记住的事情。

*看清原理*

---
