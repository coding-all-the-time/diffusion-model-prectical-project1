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
一遍过
```python
t = torch.linspace(0, T, T + 1, dtype=torch.float64)
f_t = torch.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2
alpha_bar = f_t / f_t[0]
betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
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

## train.py

### 1、读取文件
报错:
```bash
UnicodeDecodeError: 'gbk' codec can't decode byte 0x80 in position 34: illegal multibyte sequence
```
> 原因：Windows 系统在使用 Python 的 open() 函数读取文件时，默认采用的是 GBK 编码。但是，configs/mnist.yaml 文件是以 UTF-8 格式保存的，因为里面包含了中文注释。用 GBK 去强行解码 UTF-8 的文件，就会报这个错。

解决方法：在打开和保存文件位置加上encoding='utf-8'
```python
with open(args.config, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
```
```python
# 保存配置
with open(output_dir / 'config.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(cfg, f)
```

### 2、GPU配置
没有找到GPU：
```bash
D:\Anaconda\lib\site-packages\torch\amp\autocast_mode.py:198: UserWarning: User provided device_type of 'cuda', but CUDA is not available. Disabling
  warnings.warn('User provided device_type of \'cuda\', but CUDA is not available. Disabling')
```
排查错误：
```bash
PS D:\Code\machine_learning\diffusion model\diffusion-model-prectical-project1> nvidia-smi
Sun May 24 16:49:22 2026       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 546.30                 Driver Version: 546.30       CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                     TCC/WDDM  | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3060 ...  WDDM  | 00000000:01:00.0 Off |                  N/A |
| N/A   51C    P3              20W /  50W |      0MiB /  6144MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
PS D:\Code\machine_learning\diffusion model\diffusion-model-prectical-project1> python
Python 3.9.13 (main, Aug 25 2022, 23:51:50) [MSC v.1916 64 bit (AMD64)] :: Anaconda, Inc. on win32

Warning:
This Python interpreter is in a conda environment, but the environment has
not been activated.  Libraries may fail to load.  To activate this environment
please see https://conda.io/activation

Type "help", "copyright", "credits" or "license" for more information.
Ctrl click to launch VS Code Native REPL
>>> import torch
>>> print(torch.cuda.is_available())
False
```
原因是：This Python interpreter is in a conda environment, but the environment has not been activated.（Conda 环境未激活）
#### 第一步：修复并激活conda环境：
```bash
conda init powershell
```
`ctrl`+`~`重新打开终端，报错：
```bash
. : 无法加载文件 C:\Users\HZQ\Documents\WindowsPowerShell\profile.ps1，因为在此系统上禁止运行脚本。有关详细信息，请参阅 https:/go.microsoft.com/fwl
ink/?LinkID=135170 中的 about_Execution_Policies。
所在位置 行:1 字符: 3
+ . 'C:\Users\HZQ\Documents\WindowsPowerShell\profile.ps1'
+   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : SecurityError: (:) []，PSSecurityException
    + FullyQualifiedErrorId : UnauthorizedAccess
```
> 原因：运行 conda init powershell 时，Conda 修改了 PowerShell 配置文件（profile.ps1），以便每次打开终端时自动激活 Conda 环境。但是，由于 Windows 的安全策略拦截了这个脚本的运行，导致 Conda 无法成功挂载，环境无法激活，没看到 (base) 前缀。

修改执行策略为RemoteSigned：
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
重启终端生效，若要恢复执行策略，则运行：
```bash
Set-ExecutionPolicy -ExecutionPolicy Restricted -Scope CurrentUser
```

#### 第二步：安装适配的PyTorch版本
```bash
pip uninstall torch torchvision torchaudio -y
```
安装 CUDA 12.1 版本的 PyTorch
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```