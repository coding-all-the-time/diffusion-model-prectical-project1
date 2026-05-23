"""
Time embedding 模块

DDPM 的网络需要把整数时间步 t 映射成连续向量，再通过 MLP 注入到每个 ResBlock。
本文件实现 sinusoidal position embedding（与 Transformer 同款）和 TimeEmbedding 包装类。

公式（来自 L04）：
    PE(t)_{2k}   = sin( t / 10000^{2k/d} )
    PE(t)_{2k+1} = cos( t / 10000^{2k/d} )
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# TODO 7: 实现 Sinusoidal time embedding
# ============================================================================
class SinusoidalPosEmb(nn.Module):
    """
    把 (B,) 的整数时间步映射成 (B, dim) 的连续向量。

    数学：
        half = dim // 2
        freqs[k] = exp( -log(10000) * k / half ),  k = 0..half-1
        for each t:
            args = t * freqs              # (half,)
            emb  = [cos(args), sin(args)] # (dim,)

    Args:
        dim: embedding 维度（通常等于 base_channels，如 128）

    Forward:
        Input:  t shape (B,) long tensor
        Output: emb shape (B, dim) float tensor
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        assert dim % 2 == 0, f"dim must be even, got {dim}"

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) long tensor of timesteps

        Returns:
            emb: (B, dim) float tensor

        实现提示：
            1. 计算 half = dim // 2
            2. 计算频率 freqs：用 torch.exp(-math.log(10000) * arange(half) / half)
               注意把 freqs 放到 t 所在的 device 上
            3. 计算 args = t.float()[:, None] * freqs[None, :]，shape (B, half)
            4. 拼接 cos 和 sin：torch.cat([args.sin(), args.cos()], dim=-1)
        """
        # >>> 在这里写你的代码（约 5-7 行） <<<
        raise NotImplementedError("TODO 7: 实现 SinusoidalPosEmb.forward")
        # >>> 结束 <<<


# ============================================================================
# TimeEmbedding 包装：sinusoidal + 2 层 MLP（已提供，无需修改）
# ============================================================================
class TimeEmbedding(nn.Module):
    """
    完整的 timestep embedding 路径：
        t (B,) → SinusoidalPosEmb (B, dim) → MLP (B, dim*4)

    输出维度通常是 base_channels * 4（与 ResBlock 中投影层匹配）。
    """

    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4

        self.sinusoidal = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) long tensor

        Returns:
            (B, hidden_dim) float tensor
        """
        emb = self.sinusoidal(t)
        emb = self.mlp(emb)
        return emb


# ============================================================================
# 自测
# ============================================================================
if __name__ == '__main__':
    print("=== Testing embedding.py ===")

    try:
        emb_module = SinusoidalPosEmb(128)
        t = torch.tensor([0, 100, 500, 999])
        emb = emb_module(t)
        assert emb.shape == (4, 128), f"Expected (4, 128), got {emb.shape}"

        # 不同 t 的 embedding 应当不同
        assert not torch.allclose(emb[0], emb[1])

        # cos/sin 取值应当在 [-1, 1]
        assert emb.min() >= -1.0001 and emb.max() <= 1.0001

        print("✅ SinusoidalPosEmb passed")
    except NotImplementedError:
        print("⚠️  SinusoidalPosEmb not implemented (TODO 7)")

    try:
        time_emb = TimeEmbedding(128)
        t = torch.tensor([0, 100, 500, 999])
        out = time_emb(t)
        assert out.shape == (4, 512), f"Expected (4, 512), got {out.shape}"
        print("✅ TimeEmbedding passed")
    except NotImplementedError:
        print("⚠️  Skipping TimeEmbedding (depends on TODO 7)")
