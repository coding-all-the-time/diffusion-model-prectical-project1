# Experiment Log 

`exp_20260526_02_mnist_cosine_resume`

## 一句话目标

> 本次实验最想验证什么？答案应当是 yes/no 形式可判定的。

*验证 train.py 中新增的断点恢复逻辑是否能无缝衔接模型状态（Loss不突变，步数正常衔接）。补测 Cosine Schedule + fp16 在正常硬件状态下的准确训练时长（预期与第二次实验的 98.6 分钟相近）。*

---

## 1. 环境信息

| 项目 | 内容 |
|------|------|
| 实验日期 | 2026-05-26 |
| 实验者 | HZQ |
| 硬件 | NVIDIA GeForce RTX 3060 6GB × 1 |
| 框架版本 | PyTorch 2.3.0, CUDA 12.1 |
| 仓库 | github.com/coding-all-the-time/diffusion-model-prectical-project1 |
| Git commit | `9de75d4`|
| Git 分支 | `origin/main` |

---

## 2. 假设与预期

> 在跑实验**之前**写下你的预期。之后对比结果，培养科学直觉。

**假设**：
- 新增的 resume 逻辑能够正确加载 model, optimizer, ema, scaler 以及 scheduler_lr
- Cosine schedule 的计算复杂度与 Linear 几乎相同，在硬件不降频的情况下，耗时应与 baseline_2 持平

**预期结果**：
- 中断后通过 --resume 重启，训练的 Loss 应该维持在中断前的水平（例如 0.03 左右），不会突然飙升回 1.0 以上
- 中断前耗时 + 中断后耗时，总和应该在 95 ~ 105 分钟左右
- FID 分数预计维持在 2.9 左右（EMA）

**如果与预期不符可能的原因**：
- Resume 时某些随机数种子没有被保存，导致当前 batch 的数据顺序发生变化，引起微小的 Loss 波动（这是正常现象）
- 如果 Loss 发生巨大突变，可能是 Optimizer 状态或 Scaler 状态加载失败

---

## 3. 配置（完整 hyperparameters）

```yaml
output_dir: runs/exp_mnist_cosine_resume
seed: 42

# ─────────── 数据 ───────────
dataset:
  name: mnist
  batch_size: 128
  image_size: 32          
  num_workers: 4
  root: ./data

# ─────────── 模型（小型 U-Net） ───────────
model:
  in_channels: 1
  out_channels: 1
  base_channels: 64
  channel_mult: [1, 2, 2]    
  num_res_blocks: 2
  attn_resolutions: [16]      
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
  warmup_steps: 1000

# ─────────── EMA ───────────
ema_decay: 0.999

# ─────────── 混合精度（可选） ───────────
mixed_precision: fp16       

# ─────────── 训练循环 ───────────
training:
  num_epochs: 50
  log_every: 50             
  sample_every: 1000        
  ckpt_every: 5000 
  grad_clip: 1.0

# ─────────── WandB（可选） ───────────
wandb:
  enabled: true            
  project: ddpm-course
  run_name: mnist_cosine_resume

# ─────────── 绘图（可选） ───────────
plotting:
  enabled: true             
  record_every: 50          
  plot_every: 1000 
```

---

## 4. 运行命令（必须可直接复制粘贴运行）

```bash
python train.py --config configs/mnist_cosine_resume.yaml
```
运行 5000 步后，点击`ctrl+c`中断进程，再重新启动
```bash
python train.py --config configs/mnist_cosine_resume.yaml --resume runs/exp_mnist_cosine_resume/ckpt/step_005000.pt
```

---

## 5. 结果

### 5.1 数值指标

| 指标 | 数值 | 备注 |
|------|------|------|
| 训练总时长 | 114.6 min |使用140W便携适配器供电 |
| 显存峰值 | 4.12 GB | |
| 最终训练 loss | 0.02478 | |
| FID @ EMA| 2.9249 | 5000 张样本 |
| FID @ no EMA| 5.2128 | 5000 张样本 |

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
| cosine schedule/ema_decay: 0.999 (本次) | 2.9249/5.2128 | 114.6 min |exp_20260526_02，使用140W便携适配器供电 |
---

## 6. 结论

### 6.1 假设是否成立

🟡 部分成立

**简短说明**：
1. 断点恢复逻辑验证成功，Loss 在中断前后平稳衔接，未出现激增突变，最终生成质量（FID）也完全达标，证明模型权重、优化器状态及各组件调度均被正确地无缝恢复。
2. 训练耗时略高于预期，原本预期与第二次实验（约 98 分钟）持平，但实际总耗时 114.6 分钟。结合环境变量推断，本次使用 140W 便携适配器供电，显卡未能像连接原装大电源时那样发挥极限性能，导致了轻微降频拉长了时间。
3. 在正常走完 50 epochs 后，Cosine 的 EMA FID (2.9249) 与 Linear (2.8824) 不相上下，且 no EMA FID 回归到 5.2128。证明上一次实验中无 EMA 质量差是由于非正常断点和降频引起的，而非 Cosine 调度本身的问题。

### 6.2 现象记录

观察到但与预期无关的现象：

- 任务管理器显示只有核显 GPU0 在占用，但是 3060 的温度很高；wandb 也记录：system/gpu.0.enforcedPowerLimitWatts；从nvidia-smi输出来看 3060 正在使用。
> 原因：
> 1. Windows 任务管理器的 GPU 占用率默认监控的是 3D 渲染。但是，跑 Diffusion 模型使用的是 CUDA 核心进行张量计算，属于通用计算而不是图形渲染。
> 2. 在 Windows 操作系统眼里：它看到了两张显卡，按顺序把核显（Intel）叫作 GPU 0，把独显（NVIDIA）叫作 GPU 1。在 CUDA 框架和 WandB 眼里：它们只认 NVIDIA 的显卡。既然只有一张 NVIDIA 显卡，就是 GPU 0。所以，WandB 里记录的 system/gpu.0 实际上完全等同于任务管理器里的 GPU 1。

- 断点重连后，wandb 的 GPU 信息只显示连接后的，而自定义数据绑定到 step，可以显示全部。
> 原因：
> 1. loss、lr、steps_per_sec 这些数据，在 train.py 里通过 wandb.log(..., step=global_step) 显式调用。WandB 会把这些数值与 global_step 严格绑定在一起存进数据库。所以用 Step 做 X 轴时，能完美画出图。
> 2. 重新运行后，会跑一次 wandb.init()，WandB 会启动一个后台线程，每隔几秒钟去操作系统查一次硬件状态（GPU温度、内存、功耗等）。这个后台线程不知道代码跑到第几个 Step，只知道现在距离代码启动过去了多少秒（Relative Time）。

> 解决方案：对于底层的硬件监控来说，按时间来看其实是最合理的，因为即使代码卡住了没走 step，显存和温度依然在随时间发生变化，所以只需要自定义一个和 step 绑定的 GPU 占用，就可以同时看到两个数据：
```python
# ─────────── Logging ───────────
if global_step % log_every == 0:
    if use_wandb:
        wandb.log({
            'loss': loss.item(),
            'lr': lr_now,
            'epoch': epoch,
            'steps_per_sec': steps_per_sec,
            # 新增：将自己获取的显存数据传给 WandB
            'custom/gpu_mem_gb': torch.cuda.memory_reserved(device) / (1024 ** 3)
        }, step=global_step)
```

---

## 7. 失败记录与踩坑（**强制填写**）

> 没有遇到任何问题是不正常的。仔细回想是否有任何"花了 5 分钟以上才解决的事情"。

### 坑 1：断点恢复后，平均训练速度显示异常偏高（达几百step/s）
```bash
TypeError: cosine_beta_schedule() got an unexpected keyword argument 'beta_start'
```
- **现象**：执行 --resume 命令重启后，最初的几个 log 显示的训练速度极其离谱，例如：[ep 011 step 005050] loss=0.0279 lr=2.00e-04 (280.1 step/s)。随后速度缓慢下降，逐渐逼近真实的 3.4 steps/s。
- **猜测原因**：起初以为显卡算力出现异动，后来排查日志打印逻辑发现，是速度计算的分子和分母基准不匹配。
- **验证过程**：train.py 中的测速代码为 `steps_per_sec = global_step / elapsed`。Resume 时，global_step 直接从 5000 开始，但 elapsed（经过的时间）重新从 0 计时。导致程序误以为在刚启动的十几秒内跑了 5000 多步。
- **最终方法**：需要在代码启动计时前，记录启动时的步数 `initial_step = global_step`，然后将速度公式修正为只计算本次运行的实际步数：`steps_per_sec = (global_step - initial_step) / elapsed`。
- **耗时**：约 5 分钟
- **教训**：断点恢复不仅要恢复核心的模型/优化器状态，相关的辅助指标计算（如计时器、初始计步器）也必须考虑恢复场景带来的上下文突变

### 坑 2：WandB 记录断层，分为两个独立的 Run

- **现象**：断点恢复后，WandB 上生成了一个全新的 run (en5z5su5)，原先中断前的图表曲线 (1m96qfhx) 和断点后的曲线不能拼接在一起，图表被割裂
- **猜测原因**：默认情况下，每次调用 wandb.init() 都会随机生成一个新的 run_id 并将其视为一次全新的实验。代码中没有在恢复时将前一次的 run_id 传回给 WandB，也没有显式启用恢复参数。
- **最终方法**：修改 train.py
1. 在每次保存 Checkpoint .pt 文件时，提取当前 wandb.run.id 并随字典一并保存：save_dict['wandb_run_id'] = wandb.run.id。
2. 在 --resume 重启流程预加载 ckpt 时，读出旧的 ID：run_id = ckpt.get('wandb_run_id')。
3. 将取出的旧 ID 传入初始化函数：wandb.init(..., id=run_id, resume="allow")。

- **耗时**：约 30 分钟
- **教训**：如果需要云端实验记录连贯，断点恢复不须考虑到第三方 Logger / Tracking 服务的 Session ID 的提取与状态同步

---

## 8. 下一步计划

> 基于本次结果，下一个实验应该做什么？写得越具体越好。

- [x] 由于在 wandb 上后面实验的速度飙升会影响到整体的显示，所以我停止训练，把产生的两个实验删除，把本地产生的参数和采样删除，重新跑一遍验证代码。
- [ ] 挑战一下 cifar10 的训练。

---

## 9. 给未来自己的提醒

> 一句话总结这次实验最值得记住的事情。

*恢复训练不仅要还原模型状态，还要注意监控指标的数据基准是否被重置，以及云端打点服务的 Session 是否串联*

---
