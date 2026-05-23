"""
DDPM 核心算法：q_sample、p_losses、p_sample、p_sample_loop

本文件实现 DDPM 训练与采样的全部数学算法。是整个项目的核心。

请在动手实现前确保你已经：
- 完成 L03 的 q(x_t|x_0) 闭合形式手推
- 完成 L04 的 ELBO → MSE 推导
- 理解 Algorithm 1（训练）与 Algorithm 2（采样）

参考公式速查（来自 L03/L04）：
    q(x_t | x_0) = N(√ᾱ_t · x_0, (1-ᾱ_t) I)
    重参数化：x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,  ε ~ N(0,I)

    Simplified loss:
        L = E_{t, x_0, ε} [ || ε - ε_θ(x_t, t) ||² ]

    Reverse mean (DDPM):
        μ_θ = (1/√α_t) · ( x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t) )

    Sampling step:
        x_{t-1} = μ_θ + σ_t · z,    z ~ N(0,I) if t>1 else 0
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from schedule import DDPMSchedule, extract


# ============================================================================
# TODO 3: 实现 q_sample（前向加噪）
# ============================================================================
def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: DDPMSchedule,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    使用闭合形式从 x_0 一步采样到任意时间步 t 的 x_t。

    数学：x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,  ε ~ N(0,I)

    Args:
        x0: shape (B, C, H, W) 的干净样本
        t: shape (B,) 的 long tensor，每个样本的时间步（0-indexed）
        schedule: DDPMSchedule 实例
        noise: 可选，外部传入的噪声 ε。训练中通常不传，让函数内部采样。
               但 p_losses 中要保留这个 noise 用作目标，所以会传入。

    Returns:
        x_t: shape (B, C, H, W) 的加噪后样本

    提示：
        - 用 extract(schedule.sqrt_alpha_bar, t, x0.shape) 取出按 t 索引并 reshape 后的系数
        - 注意 noise 的 shape 应当与 x0 完全相同
    """
    if noise is None:
        noise = torch.randn_like(x0)

    # >>> 在这里写你的代码（约 3-4 行） <<<
    sqrt_alpha_bar = extract(schedule.sqrt_alpha_bar, t, x0.shape)
    sqrt_one_minus_alpha_bar = extract(schedule.sqrt_one_minus_alpha_bar, t, x0.shape)
    x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
    return x_t
    # >>> 结束 <<<


# ============================================================================
# TODO 4: 实现 p_losses（DDPM simplified loss）
# ============================================================================
def p_losses(
    model: nn.Module,
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: DDPMSchedule,
) -> torch.Tensor:
    """
    DDPM 的 simplified loss（Ho 2020）:
        L = || ε - ε_θ(x_t, t) ||²

    Args:
        model: 噪声预测网络 ε_θ，接受 (x_t, t) 输出与 x_t 同形状的张量
        x0: shape (B, C, H, W) 的干净样本
        t: shape (B,) 的时间步
        schedule: DDPMSchedule 实例

    Returns:
        loss: 标量 tensor

    提示：
        - 流程：采样 ε → 用 q_sample 得到 x_t → 模型预测 → 与 ε 做 MSE
        - 用 F.mse_loss(pred, target)
        - 注意要把 noise 传给 q_sample 以保持 noise 与 loss 计算的一致性
    """
    # >>> 在这里写你的代码（约 4-6 行） <<<
    noise = torch.randn_like(x0)
    xt = q_sample(x0, t, schedule, noise)
    pred_noise = model(xt, t)
    loss = F.mse_loss(pred_noise, noise)
    return loss
    # >>> 结束 <<<


# ============================================================================
# 辅助函数：从预测的噪声反推 x_0（参考实现，无需修改）
# ============================================================================
def predict_x0_from_noise(
    xt: torch.Tensor,
    t: torch.Tensor,
    pred_noise: torch.Tensor,
    schedule: DDPMSchedule,
) -> torch.Tensor:
    """
    由 x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε 反推：
        x_0 = (x_t - √(1-ᾱ_t) · ε) / √ᾱ_t

    用于可视化与某些采样器。
    """
    sqrt_alpha_bar = extract(schedule.sqrt_alpha_bar, t, xt.shape)
    sqrt_one_minus_alpha_bar = extract(schedule.sqrt_one_minus_alpha_bar, t, xt.shape)
    return (xt - sqrt_one_minus_alpha_bar * pred_noise) / sqrt_alpha_bar


# ============================================================================
# TODO 5: 实现 p_sample（单步反向采样）
# ============================================================================
@torch.no_grad()
def p_sample(
    model: nn.Module,
    xt: torch.Tensor,
    t: torch.Tensor,
    schedule: DDPMSchedule,
) -> torch.Tensor:
    """
    DDPM 单步反向采样：x_t → x_{t-1}

    数学：
        ε̂ = ε_θ(x_t, t)
        μ̂ = (1/√α_t) · ( x_t - β_t/√(1-ᾱ_t) · ε̂ )
        x_{t-1} = μ̂ + σ_t · z,    z ~ N(0,I) if t > 0 else 0

    其中 σ_t² 可选 β_t（DDPM 默认）或 \tilde β_t（posterior_variance）。
    本实现使用 \tilde β_t（理论更优，工程也更稳）。

    Args:
        model: ε_θ
        xt: shape (B, C, H, W)
        t: shape (B,) 的时间步（所有元素相同，即当前 step）
        schedule: DDPMSchedule

    Returns:
        x_{t-1}: shape (B, C, H, W)

    提示：
        - 三个关键系数：sqrt_recip_alpha, betas, sqrt_one_minus_alpha_bar
        - 用 extract 把它们 reshape 成 (B,1,1,1)
        - 噪声项的方差用 posterior_variance（也可以用 betas，差异不大）
        - 注意 t=0 时**不加随机扰动**（最后一步无需采样）
    """
    # >>> 在这里写你的代码（约 8-12 行） <<<
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
    # >>> 结束 <<<


# ============================================================================
# TODO 6: 实现完整采样循环
# ============================================================================
@torch.no_grad()
def p_sample_loop(
    model: nn.Module,
    shape: tuple,
    schedule: DDPMSchedule,
    device: torch.device,
    return_intermediates: bool = False,
) -> torch.Tensor:
    """
    从纯噪声开始迭代 T 步反向采样，得到生成样本。

    Args:
        model: ε_θ
        shape: 目标样本形状，如 (16, 3, 32, 32)
        schedule: DDPMSchedule
        device: 设备
        return_intermediates: 若为 True，额外返回所有中间 x_t（用于可视化）

    Returns:
        x_0: shape 等于 shape 的最终生成样本
        （如果 return_intermediates=True，还返回 list of intermediates）

    提示：
        - 从 x_T ~ N(0,I) 开始
        - 循环 t = T-1, T-2, ..., 0（共 T 步）
        - 每步把 x_t 喂给 p_sample 得到 x_{t-1}
        - 注意 t 是一个 (B,) 的 tensor，每个 batch 元素都填同样的 t
    """
    B = shape[0]
    T = schedule.T

    # >>> 在这里写你的代码（约 8-15 行） <<<
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
    # >>> 结束 <<<


# ============================================================================
# 自测（直接运行此文件可以验证你的实现）
# ============================================================================
if __name__ == '__main__':
    print("=== Testing diffusion.py ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    schedule = DDPMSchedule(T=1000).to(device)

    # 测试 q_sample
    try:
        x0 = torch.randn(4, 3, 32, 32, device=device)
        t = torch.tensor([0, 100, 500, 999], device=device)
        x_t = q_sample(x0, t, schedule)
        assert x_t.shape == x0.shape
        # t=999 时 x_t 应当接近纯高斯（与 x0 几乎不相关）
        # t=0 时 x_t 应当几乎等于 x0
        print(f"  q_sample at t=0: ||x_t - x0|| = {(x_t[0] - x0[0]).norm():.4f} (should be small)")
        print(f"  q_sample at t=999: var of x_t = {x_t[3].var():.4f} (should be near 1)")
        print("✅ q_sample passed")
    except NotImplementedError:
        print("⚠️  q_sample not implemented (TODO 3)")

    # 测试 p_losses（需要一个假模型）
    try:
        # 假模型：直接返回输入（仅用于测试 shape）
        class IdentityModel(nn.Module):
            def forward(self, x, t):
                return x
        fake_model = IdentityModel().to(device)

        x0 = torch.randn(4, 3, 32, 32, device=device)
        t = torch.randint(0, 1000, (4,), device=device)
        loss = p_losses(fake_model, x0, t, schedule)
        assert loss.dim() == 0, "loss should be scalar"
        print(f"  p_losses output (scalar): {loss.item():.4f}")
        print("✅ p_losses passed")
    except NotImplementedError:
        print("⚠️  p_losses not implemented (TODO 4)")

    # 测试 p_sample 与 p_sample_loop
    try:
        class DummyModel(nn.Module):
            def forward(self, x, t):
                return torch.zeros_like(x)
        dummy_model = DummyModel().to(device)

        # 用一个非常小的 schedule 测试速度
        small_schedule = DDPMSchedule(T=10).to(device)

        x_T = torch.randn(2, 3, 8, 8, device=device)
        t = torch.full((2,), 5, device=device, dtype=torch.long)
        x_prev = p_sample(dummy_model, x_T, t, small_schedule)
        assert x_prev.shape == x_T.shape
        print("✅ p_sample passed")

        samples, _ = p_sample_loop(dummy_model, (2, 3, 8, 8), small_schedule, device, True)
        assert samples.shape == (2, 3, 8, 8)
        print("✅ p_sample_loop passed")
    except NotImplementedError as e:
        print(f"⚠️  Skipped: {e}")

    print("\n如果上述全部通过，diffusion.py 实现正确！可以进入下一步训练。")
