# Project 1: From-Scratch DDPM

> **目标**：不依赖 `diffusers` 或 `lucidrains`，从零实现一个完整的 DDPM。
>
> **截止时间**：W3 周末（讲完 L04 后两周内）
>
> **技能目标**：将 L03/L04 的数学推导转化为可运行代码，培养"读完论文能复现"的能力。

## 配套教材

讲义、公式推导、论文导读在教材库，开始前请确认已 clone：

```bash
git clone https://github.com/Qi-StarterTrain/diffusion-models-starter-materials.git
```

本项目用到：`slides/L03_ddpm_forward_process.md`、`slides/L04_ddpm_training_unet_sampling.md`、以及 `derivations/` 下的 derive_02、derive_03。

---

## 学习目标

完成本项目后你应当能：
1. 解释 DDPM 训练算法每一行代码的数学对应
2. 独立写出 forward/reverse process 的完整实现
3. 调试 DDPM 训练中的常见问题（NaN、loss 不降、生成样本糊）
4. 在新数据集上快速搭建 DDPM 训练流水线

---

## 任务分档

按你的水平选择对应难度。**至少完成基础档**才能交。

### 基础档（必做，60 分）

- [ ] 实现完整代码（按照下方"代码任务清单"）
- [ ] 在 MNIST 上训练 50 epoch
- [ ] 能生成视觉可辨认的数字（手写数字应当大致清晰）
- [ ] 提交训练曲线 + 64 张样本网格 + 实验日志

### 进阶档（推荐，+20 分）

- [ ] 在 CIFAR-10 上训练 200 epoch
- [ ] FID ≤ 15（5000 样本对 5000 训练样本计算）
- [ ] 实现 EMA，对比有无 EMA 的采样质量差异
- [ ] 提交完整对比报告

### 挑战档（加分项，+20 分）

- [x] 实现 cosine schedule
- [ ] 在 CIFAR-10 上对比 linear/cosine 两种 schedule
- [ ] 至少 3 个随机种子下报告 mean ± std
- [ ] 提交 8 页技术报告，包含失败案例分析

---

## 代码任务清单（基础档必填）

下面 8 个 TODO 必须由你独立完成。`# TODO` 标记处不允许直接复制 lucidrains 的实现，但可以参考结构。

### 在 `schedule.py` 中

```python
# TODO 1: 实现 linear schedule
def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    pass  # 返回 (T,) 的 tensor

# TODO 2 (挑战档): 实现 cosine schedule
def cosine_beta_schedule(T, s=0.008):
    pass
```

### 在 `diffusion.py` 中

```python
# TODO 3: 实现 q_sample（前向加噪，使用闭合形式）
def q_sample(x0, t, sqrt_alpha_bar, sqrt_one_minus_alpha_bar, noise=None):
    pass

# TODO 4: 实现 p_losses（DDPM simplified loss）
def p_losses(model, x0, t, schedule):
    pass

# TODO 5: 实现 p_sample（单步反向采样）
@torch.no_grad()
def p_sample(model, xt, t, schedule):
    pass

# TODO 6: 实现完整采样循环
@torch.no_grad()
def p_sample_loop(model, shape, schedule):
    pass
```

### 在 `model/embedding.py` 中

```python
# TODO 7: 实现 sinusoidal time embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        pass
```

### 在 `model/unet.py` 中

```python
# TODO 8: 实现 ResBlock 中的 time embedding 注入
class ResBlock(nn.Module):
    def forward(self, x, t_emb):
        # 关键点：t_emb 通过 broadcast 加到 feature map
        pass
```

---

## 提交清单

**截止前将以下内容 push 到你的作业仓库 main 分支**，助教直接在仓库里评分。

```
（仓库根目录）
├── schedule.py / diffusion.py / model/ / ...   # 含你实现的 TODO
├── configs/mnist.yaml                           # 训练用配置
├── runs/exp_mnist_baseline/
│   ├── loss_curve.png          # 训练 loss 曲线
│   ├── samples_final.png       # 64 张生成样本网格
│   └── checkpoint_final.pt     # 模型权重（大文件请用 Git LFS）
├── logs/exp_log.md             # 至少 1 份实验日志（按 experiment_log_template.md）
├── report.md                   # 1-2 页报告
└── debug_log.md                # 至少 3 条踩坑记录
```

**实验日志 / 踩坑记录格式**：见 `templates/experiment_log_template.md`。

**报告内容要求**：
1. 训练曲线 + 样本图
2. 关键超参数列表（train batch size、lr、T、schedule type 等）
3. 至少回答下面"自查问题"前 5 道
4. 至少一项"我学到的最有价值的 1 件事"

---

## 自查问题（在报告中回答）

下面问题旨在确认你**真的懂了**，不是抄了一份代码就过：

1. **公式映射**：解释 `q_sample` 中 `sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise` 对应论文哪个公式？
2. **形状广播**：为什么从 `sqrt_alpha_bar` 的 (T,) tensor 取出 batch 的 `t` 后要 reshape 成 (B, 1, 1, 1)？不这样会怎样？
3. **设备**：为什么 `betas` 必须用 `register_buffer` 而不是 `self.betas = ...`？
4. **训练**：为什么每个 step 的 `t` 是随机采样而不是顺序遍历？
5. **采样**：`p_sample` 的最后一步（`t=0`）为什么不加随机扰动？
6. **EMA**（进阶档）：实测下来用 EMA 和不用，FID 差多少？为什么会有这个差异？
7. **Schedule**（挑战档）：linear 和 cosine 在 50 epoch 时哪个更好？200 epoch 时呢？解释原因。

---

## 评分标准

| 维度 | 满分 | 说明 |
|------|------|------|
| 代码正确性 | 30 | 8 个 TODO 全部正确实现，无逻辑错误 |
| 训练效果 | 20 | MNIST 至少能生成可辨认数字 |
| 实验日志 | 15 | 按模板填写，包含完整超参与结果 |
| 踩坑记录 | 10 | ≥3 条真实踩坑，分析到位 |
| 报告质量 | 15 | 自查问题回答正确、深入 |
| 代码质量 | 10 | 注释、结构清晰可读 |
| **加分** | +40 | 进阶档 +20 / 挑战档 +20 |

---

## 关键提醒

⚠️ **学术诚信**

- 允许参考开源实现（lucidrains、diffusers 等）的**结构**
- 但每个 TODO 必须自己写、自己调
- 直接复制粘贴并提交将记 0 分
- 报告中的"踩坑记录"必须真实——抄来的踩坑是看得出来的

⚠️ **环境**

- PyTorch 2.0+，CUDA 11.8+
- 单卡 RTX 3060 12GB 可完成基础档（MNIST）
- 单卡 RTX 3090/4090 24GB 可完成进阶档（CIFAR-10）
- 没有显卡？用 Colab Pro 或 AutoDL 的入门级实例

⚠️ **常见错误（看完再开始）**

1. **数据归一化**：MNIST 默认 `[0,1]`，需归一化到 `[-1,1]`：`x = x * 2 - 1`
2. **t 索引**：代码里 t 通常从 0 开始（0-indexed），论文公式从 1 开始
3. **schedule 设备**：忘记 `register_buffer` 是新手 90% 的 bug 来源
4. **fp16 NaN**：第一次训练务必用 fp32，确认正常后再开混合精度

---

## 时间规划建议

| 时间 | 任务 |
|------|------|
| Day 1 | 阅读 L03、L04，手推全部公式 |
| Day 2 | 实现 schedule.py + diffusion.py |
| Day 3 | 实现 model/（U-Net + embedding） |
| Day 4 | 在 MNIST 训练，调参，可视化 |
| Day 5 | （进阶档）迁移到 CIFAR-10 |
| Day 6-7 | （挑战档）schedule 对比实验 + 报告撰写 |

不要拖到最后一天。Diffusion 模型训练慢，CIFAR-10 跑 200 epoch 要 8+ 小时。

---

## 帮助资源

- **遇到 bug**：先看 `debugging_checklist.md`
- **手推不会**：回到 derive_02、derive_03
- **代码思路**：可参考 lucidrains 实现，但**禁止逐行复制**
- **训练慢**：检查 num_workers、pin_memory、batch_size、混合精度

---

> **写在最后**：这个 Project 是整个课程的基石。从零写一遍 DDPM 的过程，比看 100 篇综述都更能让你"真正懂 DDPM"。这份理解会贯穿后续所有内容。
>
> 完成后回头看，你会发现 DDPM 的代码极其简短——这就是数学之美。
