"""
DDPM Noise Schedule 模块

本文件实现两类 noise schedule（linear / cosine），并将所有由 schedule 推出的
量（alpha、alpha_bar、sqrt_alpha_bar 等）打包成一个 nn.Module。

⚠️ 重要：所有 schedule 张量必须用 register_buffer 注册，否则 .to(device) 时
   不会跟着移动。这是新手最常踩的坑之一。

参考公式（来自 L03）：
    α_t = 1 - β_t
    ᾱ_t = ∏_{s=1}^t α_s

调用示例：
    schedule = DDPMSchedule(T=1000, beta_schedule='linear').to(device)
    sqrt_alpha_bar_t = schedule.sqrt_alpha_bar[t]
"""

import math
import torch
import torch.nn as nn


# ============================================================================
# TODO 1: 实现 linear beta schedule
# ============================================================================
def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """
    DDPM 原始论文的 linear schedule。

    Args:
        T: 总扩散步数（例如 1000）
        beta_start: 起始 beta 值
        beta_end: 结束 beta 值

    Returns:
        betas: shape (T,) 的 1D tensor，betas[0]=beta_start, betas[T-1]=beta_end

    提示：
        - 用 torch.linspace 一行就能搞定
        - 注意 dtype，建议用 torch.float64 以提升数值精度
    """
    # >>> 在这里写你的代码（约 1 行） <<<
    raise NotImplementedError("TODO 1: 实现 linear_beta_schedule")
    # >>> 结束 <<<


# ============================================================================
# TODO 2: 实现 cosine beta schedule（Improved DDPM）
# ============================================================================
def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """
    Nichol & Dhariwal 2021 Improved DDPM 的 cosine schedule。

    思路：
        f(t) = cos²((t/T + s) / (1 + s) * π/2)
        ᾱ_t = f(t) / f(0)
        β_t = 1 - ᾱ_t / ᾱ_{t-1}

    Args:
        T: 总扩散步数
        s: 防止 t=0 时数值奇异的小偏移量

    Returns:
        betas: shape (T,) 的 1D tensor

    提示：
        - 先构造 ᾱ_t，再反推 β_t
        - 用 torch.clip 把 β 限制到 (1e-5, 0.999) 内防止数值问题
    """
    # >>> 在这里写你的代码（约 5-8 行） <<<
    raise NotImplementedError("TODO 2 (挑战档): 实现 cosine_beta_schedule")
    # >>> 结束 <<<


# ============================================================================
# Schedule 类（已提供完整实现，理解后即可使用）
# ============================================================================
class DDPMSchedule(nn.Module):
    """
    打包所有 schedule-derived 量。

    Attributes (全部以 register_buffer 注册):
        betas:                   β_t,                shape (T,)
        alphas:                  α_t = 1 - β_t,      shape (T,)
        alphas_cumprod:          ᾱ_t,                shape (T,)
        alphas_cumprod_prev:     ᾱ_{t-1}, t≥1,       shape (T,)
        sqrt_alpha_bar:          √ᾱ_t,               shape (T,)
        sqrt_one_minus_alpha_bar:√(1-ᾱ_t),          shape (T,)
        sqrt_recip_alpha:        1/√α_t,             shape (T,)
        posterior_variance:      \tilde β_t,         shape (T,)
        posterior_log_variance:  log(\tilde β_t),    shape (T,) (clipped)
        posterior_mean_coef1:    用于 \tilde μ 的 x_0 系数
        posterior_mean_coef2:    用于 \tilde μ 的 x_t 系数
    """

    def __init__(self, T: int = 1000, beta_schedule: str = 'linear', **kwargs):
        super().__init__()
        self.T = T

        # 1. 选择 schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(T, **kwargs)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(T, **kwargs)
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule}")

        # 2. 计算所有 derived 量
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # ᾱ_t
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=alphas_cumprod.dtype),
                                          alphas_cumprod[:-1]])

        sqrt_alpha_bar = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alpha = 1.0 / torch.sqrt(alphas)

        # posterior q(x_{t-1} | x_t, x_0) 的方差与均值系数
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # 防止 t=0 时 log(0)
        posterior_log_variance = torch.log(torch.clip(posterior_variance, min=1e-20))

        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

        # 3. 全部 register_buffer（关键！）
        # 用 .float() 转为 fp32（cumprod 用 fp64 计算，存储用 fp32）
        for name, tensor in [
            ('betas', betas.float()),
            ('alphas', alphas.float()),
            ('alphas_cumprod', alphas_cumprod.float()),
            ('alphas_cumprod_prev', alphas_cumprod_prev.float()),
            ('sqrt_alpha_bar', sqrt_alpha_bar.float()),
            ('sqrt_one_minus_alpha_bar', sqrt_one_minus_alpha_bar.float()),
            ('sqrt_recip_alpha', sqrt_recip_alpha.float()),
            ('posterior_variance', posterior_variance.float()),
            ('posterior_log_variance', posterior_log_variance.float()),
            ('posterior_mean_coef1', posterior_mean_coef1.float()),
            ('posterior_mean_coef2', posterior_mean_coef2.float()),
        ]:
            self.register_buffer(name, tensor)


# ============================================================================
# 工具函数：按 t 索引并 reshape 用于广播
# ============================================================================
def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    从 schedule 张量 a 中按 t 索引，并 reshape 到适合与 x 广播的形状。

    Args:
        a: (T,) 的 schedule 张量，例如 sqrt_alpha_bar
        t: (B,) 的 long tensor，每个样本的时间步
        x_shape: 目标 broadcast 形状（一般是 x.shape）

    Returns:
        (B, 1, 1, ..., 1) 的张量（额外维度数与 x 匹配）

    示例：
        x.shape = (B, 3, 32, 32) → 返回 shape (B, 1, 1, 1)
    """
    B = t.shape[0]
    out = a.gather(0, t)  # (B,)
    return out.reshape(B, *([1] * (len(x_shape) - 1)))


# ============================================================================
# 自测（直接运行此文件可以验证你的实现）
# ============================================================================
if __name__ == '__main__':
    print("=== Testing schedule.py ===")

    # 测试 linear schedule
    try:
        T = 1000
        betas = linear_beta_schedule(T)
        assert betas.shape == (T,), f"Expected shape ({T},), got {betas.shape}"
        assert abs(betas[0].item() - 1e-4) < 1e-6, f"betas[0] should be 1e-4, got {betas[0]}"
        assert abs(betas[-1].item() - 0.02) < 1e-6, f"betas[-1] should be 0.02, got {betas[-1]}"
        print("✅ linear_beta_schedule passed")
    except NotImplementedError:
        print("⚠️  linear_beta_schedule not implemented yet (TODO 1)")

    # 测试 DDPMSchedule（依赖 linear）
    try:
        schedule = DDPMSchedule(T=1000, beta_schedule='linear')

        # 完整性检查
        assert schedule.alphas_cumprod[0].item() < 1.0
        assert schedule.alphas_cumprod[-1].item() < 0.01, "ᾱ_T 应该接近 0"
        assert schedule.sqrt_alpha_bar.shape == (1000,)
        print("✅ DDPMSchedule (linear) passed")

        # extract 测试
        t = torch.tensor([0, 100, 500, 999])
        x_dummy = torch.zeros(4, 3, 32, 32)
        out = extract(schedule.sqrt_alpha_bar, t, x_dummy.shape)
        assert out.shape == (4, 1, 1, 1)
        print("✅ extract passed")
    except NotImplementedError:
        print("⚠️  Skipping DDPMSchedule test (linear schedule not ready)")

    # 测试 cosine（挑战档可选）
    try:
        betas_cos = cosine_beta_schedule(1000)
        assert betas_cos.shape == (1000,)
        print("✅ cosine_beta_schedule passed (挑战档)")
    except NotImplementedError:
        print("⚠️  cosine_beta_schedule not implemented (TODO 2, 挑战档可选)")

    print("\n如果上述全部通过，schedule.py 实现正确！")
