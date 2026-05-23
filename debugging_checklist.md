# DDPM Debugging Checklist

> 训练 DDPM 时遇到问题，按本清单排查。**90% 的 bug 都在前 5 个常见原因里**。
>
> 阅读建议：先把"通用排查流程"过一遍，再按症状找对应章节。

---

## 通用排查流程（任何问题都先做）

### Step 1：先确认这不是"看起来错但其实对"

- 训练前 1000 步 loss 在 [0.5, 1.0] 之间是**正常**的（噪声预测的初始量级）
- loss 收敛到 ~0.02 后基本不再下降也是**正常**的（不是 bug）
- 前 10 epoch 生成样本是糊的也是**正常**的（CIFAR-10 至少要 50 epoch）

如果你的"问题"属于上述任何一种，请耐心等待。

### Step 2：跑通 self-test

每个核心模块都有 `if __name__ == '__main__'` 自测。先确保：
```bash
python schedule.py    # 应输出 ✅ 三行
python diffusion.py   # 应输出 ✅ 四行
python -m model.embedding  # 应输出 ✅ 两行
python -m model.unet       # 应输出 ✅ 两行
python dataset.py     # 应输出 ✅ 两行
```

如果任何一个不通过，先修这个模块再说。

### Step 3：用最小数据量验证训练能跑通

用 batch_size=4，跑 100 步看 loss 是否下降。

```bash
# 临时改小配置
python train.py --config configs/mnist.yaml --output_dir runs/sanity_check
```

如果 100 步 loss 没下降，说明梯度根本没流。

---

## 症状 1：训练 loss 不下降 / 收敛太慢

### 排查清单

#### ☐ 1.1 数据归一化错了

DDPM 期望 $x_0 \in [-1, 1]$。若数据在 $[0, 1]$，前向加噪后均值不为 0，模型学得很差。

**检查**：
```python
x = next(iter(loader))
print(f"data range: [{x.min():.3f}, {x.max():.3f}]")
# 应该输出大约 [-1.000, 1.000]
```

如果是 `[0, 1]`，回到 `dataset.py` 检查 `Normalize` transform。

#### ☐ 1.2 schedule 张量没在 GPU 上

```python
schedule = DDPMSchedule(T=1000)  # 在 CPU
schedule = schedule.to(device)    # 必须 .to(device)！
```

如果你忘了用 `register_buffer`，`.to(device)` 不会移动 schedule 张量，会爆出非常奇怪的错（CPU/GPU 混合错误，或者 silent fail）。

#### ☐ 1.3 time embedding 没注入

打开 `model/unet.py` 中你实现的 `ResBlock.forward`，确认有这一行：
```python
h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
```

没有这一行，模型对 `t` 完全没有感知，会变成"对所有 t 输出同一个噪声"。

**测试**：固定 `x_t`，给不同 `t`，模型输出应当不同：
```python
x = torch.randn(1, 3, 32, 32, device=device)
out_t0 = model(x, torch.tensor([0], device=device))
out_t999 = model(x, torch.tensor([999], device=device))
print(f"diff: {(out_t0 - out_t999).abs().mean():.4f}")
# 应当 > 0.05；如果接近 0，time embedding 失效
```

#### ☐ 1.4 学习率不对

- 太大（>1e-3）：loss 震荡或爆炸
- 太小（<1e-5）：训练极慢

DDPM 标配是 `lr=2e-4`，配 warmup。

#### ☐ 1.5 batch size 太小

batch_size < 16 的训练动力学会很差。如果显存不够，开 fp16 或减小模型。

---

## 症状 2：loss 出现 NaN / Inf

### 排查清单

#### ☐ 2.1 fp16 GroupNorm 下溢

最常见原因。GroupNorm 在 fp16 下计算方差会下溢。

**临时解决**：先用 fp32 跑，确认没其他 bug。
```yaml
mixed_precision: 'no'
```

**长期解决**：用 bf16（A100/H100 支持），或在 GroupNorm 前后强制 autocast(False)。

#### ☐ 2.2 cosine schedule 边界 β 过大

如果你实现的 `cosine_beta_schedule` 没有 clip，可能产生 β > 0.999 的值，破坏数值稳定性。

**修复**：
```python
betas = torch.clip(betas, min=1e-5, max=0.999)
```

#### ☐ 2.3 梯度爆炸

确认 `train.py` 中开了 grad clip：
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### ☐ 2.4 数据集中有异常样本

某些 CIFAR-10 重新打包版本会有黑图或全 1 图，导致归一化后某个 batch 异常。

**检查**：
```python
for x in loader:
    assert torch.isfinite(x).all(), "data has NaN/Inf"
```

---

## 症状 3：生成样本是纯噪声

### 排查清单

#### ☐ 3.1 采样方向反了

正确的方向是 `t = T-1, T-2, ..., 0`。如果你写成 `range(T)`，相当于"先最后一步、再倒数第二步……"，方向反了。

**检查 `p_sample_loop`**：
```python
for t_step in reversed(range(T)):  # 从 T-1 到 0
    t = torch.full((B,), t_step, device=device, dtype=torch.long)
    x = p_sample(model, x, t, schedule)
```

#### ☐ 3.2 t 索引混淆

代码里 `t` 是 0-indexed（取值 0..T-1），论文公式是 1-indexed（取值 1..T）。两套要保持一致。

最简单原则：**代码全程 0-indexed**，并把"论文中的 t-1"对应到"代码中的 t"。

#### ☐ 3.3 模型完全没训练好

跑了多少 step？MNIST 至少 5000 step，CIFAR-10 至少 50K step 才能生成可辨认样本。

#### ☐ 3.4 schedule 系数计算错了

打印中间值检查：
```python
schedule = DDPMSchedule(T=1000)
print(f"ᾱ_0 = {schedule.alphas_cumprod[0]:.6f}  (should be ~0.9999)")
print(f"ᾱ_500 = {schedule.alphas_cumprod[500]:.6f}  (should be ~0.01)")
print(f"ᾱ_999 = {schedule.alphas_cumprod[-1]:.6e}  (should be ~1e-5)")
```

---

## 症状 4：生成样本糊 / 缺少细节

不一定是 bug，可能是质量瓶颈。但先排除：

#### ☐ 4.1 用了非 EMA 权重

EMA 权重几乎总是显著优于原始权重。检查你的采样脚本：
```bash
python sample.py --ckpt xxx.pt          # 默认用 EMA ✅
python sample.py --ckpt xxx.pt --no_ema # 不用 EMA（仅对比测试）
```

#### ☐ 4.2 采样步数过少

DDPM 标准是 T 步采样。你不能在采样时跳步——那是 DDIM 的事（W6 内容）。

#### ☐ 4.3 训练 epoch 不够

CIFAR-10 想达到 FID < 15 至少 200 epoch（约 80K step）。

#### ☐ 4.4 模型容量不够

如果用了 base_channels=32 的小模型训 CIFAR-10，FID 会很难低于 30。

---

## 症状 5：CUDA OOM（显存不足）

按下面顺序逐个尝试：

1. **batch_size 减半**：从 128 → 64 → 32
2. **开 fp16**：`mixed_precision: fp16`
3. **减小模型**：base_channels 减半（128 → 64）
4. **gradient checkpointing**（牺牲速度换显存）
5. 如果还是不够，换更小数据集（MNIST）或租 GPU

---

## 症状 6：训练慢得离谱（< 1 step/s）

#### ☐ 6.1 num_workers=0

```yaml
dataset:
  num_workers: 4   # 至少 4，不要 0
```

#### ☐ 6.2 没开 pin_memory

代码里 `pin_memory=True`（默认开了）。

#### ☐ 6.3 模型在 CPU 上

```python
print(next(model.parameters()).device)  # 应当是 cuda:0
```

#### ☐ 6.4 schedule 在 CPU 上

```python
print(schedule.betas.device)  # 应当是 cuda:0
```

每步都要把数据从 CPU 同步到 GPU 会非常慢。

#### ☐ 6.5 调用了昂贵的同步操作

`tensor.item()`、`tensor.cpu()`、`print(loss)` 都会强制 GPU 同步。每步只在 log_every 时才做。

---

## 症状 7：FID 计算异常高（>100）

#### ☐ 7.1 样本数太少

FID < 1000 样本时方差极大。**至少用 5000 样本**。

#### ☐ 7.2 样本和真实数据的分辨率/通道不一致

torchmetrics 的 FID 期望 RGB 3 通道、uint8 [0,255]。MNIST 灰度需要 repeat 成 3 通道（`evaluate.py` 已处理）。

#### ☐ 7.3 没用 EMA 权重

加 `--no_ema` 看差异，正常情况下 EMA 比 raw 低 2–5 个 FID 点。

---

## 高级调试：观察中间过程

如果上面都排查过仍有问题，可视化中间状态：

```python
# 1. 可视化 forward process
x0 = next(iter(loader))[0:1].to(device)
fig, axes = plt.subplots(1, 11, figsize=(22, 2))
for i, t_val in enumerate([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]):
    t = torch.tensor([t_val], device=device)
    xt = q_sample(x0, t, schedule)
    axes[i].imshow(denormalize(xt[0]).cpu().permute(1, 2, 0))
    axes[i].set_title(f't={t_val}')
plt.savefig('forward_process.png')
```

正确的输出：
- t=0：原图
- t=200：仍可辨认但有噪
- t=500：模糊但有结构
- t=999：纯噪声

如果 t=200 已经是纯噪声，说明 schedule 的 β 太大。

```python
# 2. 可视化反向过程的中间步（修改 p_sample_loop 加 return_intermediates）
samples, intermediates = p_sample_loop(model, shape, schedule, device, return_intermediates=True)
# 每隔 100 步保存一次 intermediate，看是否真的逐步去噪
```

---

## 终极武器：与开源实现对比

如果实在排查不出，与 `lucidrains/denoising-diffusion-pytorch` 同时用相同 seed 跑一遍 sanity check。

但**不要直接复制代码**——逐步对照你的实现差异，找到 bug 后**自己修**。这个调试过程本身就是学习。

---

## 提交前自查（开始训练前过一遍）

- [ ] 5 个 self-test 全部通过
- [ ] schedule 张量在 device 上（`schedule.betas.device == device`）
- [ ] 数据范围是 [-1, 1]（打印过 `x.min(), x.max()`）
- [ ] time embedding 真的注入了（不同 t 输出不同）
- [ ] 配置中 `output_dir` 不会覆盖之前的实验
- [ ] WandB project name 设置正确（如果用的话）
- [ ] `seed` 固定了（便于复现）

---

## 给未来自己的提醒

99% 的 DDPM bug 都属于以下五类（按出现频率）：

1. ⭐⭐⭐⭐⭐ schedule 没用 register_buffer
2. ⭐⭐⭐⭐ 数据归一化忘了 `[0,1] → [-1,1]`
3. ⭐⭐⭐⭐ time embedding 在 ResBlock 中没注入
4. ⭐⭐⭐ 采样方向写反
5. ⭐⭐⭐ fp16 + GroupNorm 出 NaN

把这五条贴在工位上，你会发现训练扩散模型其实没那么难。
