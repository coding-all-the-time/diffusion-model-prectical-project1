# debug log

## schedule.py

### TODO 1

```python
# 法一：
# return torch.linspace(beta_start, beta_end, T, dtype=torch.float64)

# 法二：
betas  = torch.tensor([beta_start + (beta_end - beta_start) * (t - 1) / (T - 1) for t in range(T)], dtype=torch.float64)
return betas
```
法二报错：
```bash
Traceback (most recent call last):
  File "d:\Code\machine_learning\diffusion model\diffusion-model-prectical-project1\schedule.py", line 184, in <module>
    assert abs(betas[0].item() - 1e-4) < 1e-6, f"betas[0] should be 1e-4, got {betas[0]}"
AssertionError: betas[0] should be 1e-4, got 8.008008008008008e-05
```
> 原因：range(T)的范围是[0,T-1],分子应该是t，否则会出现负数。

```python
# 法一：
# return torch.linspace(beta_start, beta_end, T, dtype=torch.float64)

# 法二：
betas  = torch.tensor([beta_start + (beta_end - beta_start) * t / (T - 1) for t in range(T)], dtype=torch.float64)
return betas
```

### TODO 2
```python
t = torch.arange(T, dtype=torch.float64)
alpha_bar = torch.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2
alpha_bar = torch.clip(alpha_bar, 0, 1)
betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
betas = torch.clip(betas, 1e-5, 0.999)
return betas
```
报错:
```bash
Traceback (most recent call last):
  File "d:\Code\machine_learning\diffusion model\diffusion-model-prectical-project1\schedule.py", line 217, in <module>
    assert betas_cos.shape == (1000,)
AssertionError
```
> 原因：beta的个数不等于1000，第四行得出的betas只有T - 1个，缺少alpha_bar[0]的情况
```python
t = torch.arange(T, dtype=torch.float64)
alpha_bar = torch.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2
alpha_bar = torch.clip(alpha_bar, 0, 1)
betas = torch.zeros(T, dtype=torch.float64)
betas[0] = 1 - alpha_bar[0] / 1.0
betas[1:] = 1 - (alpha_bar[1:] / alpha_bar[:-1])
betas = torch.clip(betas, 1e-5, 0.999)
return betas
```

## diffusion.py

### TODO 3

一遍过
```python
sqrt_alpha_bar = extract(schedule.sqrt_alpha_bar, t, x0.shape)
sqrt_one_minus_alpha_bar = extract(schedule.sqrt_one_minus_alpha_bar, t, x0.shape)
x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
return x_t
```

### TODO 4
一遍过
```python
noise = torch.randn_like(x0)
xt = q_sample(x0, t, schedule, noise)
pred_noise = model(xt, t)
loss = F.mse_loss(pred_noise, noise)
return loss
```

### TODO 5
一遍过
```python
sqrt_recip_alpha = extract(schedule.sqrt_recip_alpha, t, xt.shape)
betas = extract(schedule.betas, t, xt.shape)
sqrt_one_minus_alpha_bar = extract(schedule.sqrt_one_minus_alpha_bar, t, xt.shape)
pred_noise = model(xt, t)
mu = sqrt_recip_alpha * (xt - betas * pred_noise / sqrt_one_minus_alpha_bar)
if t[0] > 0:
    posterior_variance = extract(schedule.posterior_variance, t, xt.shape)
    noise = torch.randn_like(xt)
    x_prev = mu + torch.sqrt(posterior_variance) * noise
else:
    x_prev = mu
return x_prev
```

> AI补漏：虽然在你的 p_sample_loop 中，整个 batch 的 $t$ 都是同步且相同的，但 p_sample 作为一个通用函数，完全可能会接收到 不同步的时间步张量（例如：在动态 batch 生成、某些高级采样加速算法或蒸馏算法中，不同样本可能处于不同的 $t$）。直接用 t[0] 代表整个 batch 会导致严重的逻辑错误——只要 batch 第一个样本的 $t > 0$，其他即使 $t = 0$ 的样本也会被错误地注入噪声。

修复方案：消除 if/else 分支，利用 mask 进行向量化计算，让每个样本独立判断是否加噪：
```python
sqrt_recip_alpha = extract(schedule.sqrt_recip_alpha, t, xt.shape)
betas = extract(schedule.betas, t, xt.shape)
sqrt_one_minus_alpha_bar = extract(schedule.sqrt_one_minus_alpha_bar, t, xt.shape)
pred_noise = model(xt, t)
mu = sqrt_recip_alpha * (xt - betas * pred_noise / sqrt_one_minus_alpha_bar)
posterior_variance = extract(schedule.posterior_variance, t, xt.shape)
noise = torch.randn_like(xt)
nonzero_mask = (t > 0).float().view(-1, *([1] * (len(xt.shape) - 1))) 
x_prev = mu + nonzero_mask * torch.sqrt(posterior_variance) * noise
return x_prev
```


### TODO 6

一遍过

```python
x_t = torch.randn(shape, device=device)
intermediates = [x_t] if return_intermediates else None
for t in reversed(range(T)):
    t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
    x_t = p_sample(model, x_t, t_tensor, schedule)
    if return_intermediates:
        intermediates.append(x_t)
if return_intermediates:
    return x_t, intermediates
return x_t
```

验证代码中没有验证return_intermediates的功能，当我加入最后一个参数为True时，报错：
```bash
  File "d:\Code\machine_learning\diffusion model\diffusion-model-prectical-project1\diffusion.py", line 283, in <module>
    assert samples.shape == (2, 3, 8, 8)
AttributeError: 'tuple' object has no attribute 'shape'
```

> 原因：函数返回两个参数，接收都存到samples中了。
```python
samples, _ = p_sample_loop(dummy_model, (2, 3, 8, 8), small_schedule, device, True)
```

> AI补漏：扩散模型通常需要 $T=1000$ 步。$x_t$ 是完整分辨率的 GPU 张量。假设生成的是 Batch Size=16 的 $64 \times 64$ RGB 图像，单步张量约为 786 KB。收集 1000 步直接占用约 786 MB 显存。如果是 $256 \times 256$ 的图像，需要近 12.5 GB 的显存，容易崩溃。

需要将列表转到cpu中内存中：
```python
x_t = torch.randn(shape, device=device)
intermediates = [x_t.cpu()] if return_intermediates else None
for t in reversed(range(T)):
    t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
    x_t = p_sample(model, x_t, t_tensor, schedule)
    if return_intermediates:
        intermediates.append(x_t.cpu())
if return_intermediates:
    return x_t, intermediates
return x_t
```

## embedding.py
### TODO 7
一遍过
```python
half = self.dim // 2
freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
args = t.float()[:, None] * freqs[None, :]
emb = torch.cat([args.sin(), args.cos()], dim=-1)
return emb
```

### TODO 8
一遍过
```python
h = self.conv1(F.silu(self.norm1(x)))
t_proj = self.time_mlp(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
h = h + t_proj
h = self.conv2(self.dropout(F.silu(self.norm2(h))))
return h + self.skip(x)
```