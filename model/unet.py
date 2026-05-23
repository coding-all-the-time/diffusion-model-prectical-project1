"""
DDPM U-Net 架构

参考 DDPM (Ho et al. 2020) 的 U-Net 设计。

架构图：
    Input (B, C_in, H, W)
        ↓
    Conv (C_in → ch)
        ↓
    [Down stages]
        ResBlock × num_res_blocks
        (Attention if resolution in attn_resolutions)
        ─── skip ──┐
        Downsample  │
        ↓          ↓
    [Mid (Bottleneck)]
        ResBlock + Attention + ResBlock
        ↓
    [Up stages]
        Concat skip + ResBlock × (num_res_blocks + 1)
        (Attention if resolution in attn_resolutions)
        Upsample
        ↓
    GroupNorm + SiLU + Conv (ch → C_out)
        ↓
    Output (B, C_out, H, W)

时间 embedding 通过广播加法注入到每个 ResBlock。
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding import TimeEmbedding


# ============================================================================
# TODO 8: 在 ResBlock.forward 中实现 time embedding 注入
# ============================================================================
class ResBlock(nn.Module):
    """
    带 time embedding 的 ResBlock。

    结构：
        x ─→ GN → SiLU → Conv ─→ + (time_emb projected) ─→ GN → SiLU → Dropout → Conv ─→ + skip

    Args:
        in_ch: 输入通道
        out_ch: 输出通道
        time_dim: time embedding 维度
        dropout: dropout 概率
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(32, in_ch), num_channels=in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        # 把 time embedding 投影到 out_ch 通道
        self.time_mlp = nn.Linear(time_dim, out_ch)

        self.norm2 = nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # skip connection（如果通道数变化，用 1x1 conv 对齐）
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_ch, H, W) feature map
            t_emb: (B, time_dim) time embedding

        Returns:
            (B, out_ch, H, W) feature map

        实现步骤（TODO 8）:
            1. h = conv1(silu(norm1(x)))
            2. 把 t_emb 通过 time_mlp 投影到 out_ch 维：t_proj = time_mlp(silu(t_emb))
               然后 reshape 成 (B, out_ch, 1, 1) 以便广播加法
            3. h = h + t_proj  ← 这是 time injection 的关键！
            4. h = conv2(dropout(silu(norm2(h))))
            5. return h + skip(x)
        """
        # >>> 在这里写你的代码（约 5-7 行） <<<
        h = self.conv1(F.silu(self.norm1(x)))
        t_proj = self.time_mlp(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = h + t_proj
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)
        # >>> 结束 <<<


# ============================================================================
# Self-Attention（已提供，无需修改）
# ============================================================================
class SelfAttention(nn.Module):
    """
    标准 self-attention，作用在空间维度上。
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        # (B, C, H, W) → (B, num_heads, head_dim, H*W)
        q = q.reshape(B, self.num_heads, self.head_dim, H * W)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W)

        attn = torch.einsum('bhdi,bhdj->bhij', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhij,bhdj->bhdi', attn, v)

        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


# ============================================================================
# Down / Up sampling（已提供，无需修改）
# ============================================================================
class Downsample(nn.Module):
    """步长 2 的卷积下采样。"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """最近邻上采样 + 卷积。"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# ============================================================================
# UNet 主体（已提供完整实现，理解后即可使用）
# ============================================================================
class UNet(nn.Module):
    """
    DDPM 风格的 U-Net。

    Args:
        in_channels: 输入通道（RGB=3，灰度=1）
        out_channels: 输出通道（一般等于 in_channels）
        base_channels: 第一层通道数（一般 64 / 128 / 192）
        channel_mult: 各 stage 的通道倍数，如 [1, 2, 2, 2]
        num_res_blocks: 每个 stage 的 ResBlock 数量
        attn_resolutions: 在哪些分辨率（H 值）添加 self-attention
        dropout: dropout 概率
        time_emb_dim: time embedding 输出维度（一般 = base_channels * 4）

    示例（CIFAR-10）：
        UNet(in_channels=3, out_channels=3, base_channels=128,
             channel_mult=[1, 2, 2, 2], num_res_blocks=2, attn_resolutions=[16])
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: List[int] = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attn_resolutions: Tuple[int, ...] = (16,),
        dropout: float = 0.1,
        time_emb_dim: int = None,
        image_size: int = 32,
    ):
        super().__init__()

        if time_emb_dim is None:
            time_emb_dim = base_channels * 4

        # Time embedding
        self.time_embedding = TimeEmbedding(base_channels, time_emb_dim)

        # Initial conv
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # ─────────── Encoder ───────────
        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        ch = base_channels
        skip_channels = [ch]  # 记录每个 stage 末尾的通道，用于 decoder
        current_res = image_size

        for i, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch, out_ch, time_emb_dim, dropout))
                if current_res in attn_resolutions:
                    attns.append(SelfAttention(out_ch))
                else:
                    attns.append(nn.Identity())
                ch = out_ch
                skip_channels.append(ch)
            self.down_blocks.append(blocks)
            self.down_attns.append(attns)

            # 最后一个 stage 不下采样
            if i < len(channel_mult) - 1:
                self.downsamplers.append(Downsample(ch))
                skip_channels.append(ch)
                current_res //= 2
            else:
                self.downsamplers.append(nn.Identity())

        # ─────────── Bottleneck ───────────
        self.mid_block1 = ResBlock(ch, ch, time_emb_dim, dropout)
        self.mid_attn = SelfAttention(ch)
        self.mid_block2 = ResBlock(ch, ch, time_emb_dim, dropout)

        # ─────────── Decoder ───────────
        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for j in range(num_res_blocks + 1):  # +1 是为了消化 skip
                # 输入通道 = 当前 ch + skip 通道
                skip_ch = skip_channels.pop()
                blocks.append(ResBlock(ch + skip_ch, out_ch, time_emb_dim, dropout))
                if current_res in attn_resolutions:
                    attns.append(SelfAttention(out_ch))
                else:
                    attns.append(nn.Identity())
                ch = out_ch
            self.up_blocks.append(blocks)
            self.up_attns.append(attns)

            # 第一个（即最深的）stage 不上采样
            if i > 0:
                self.upsamplers.append(Upsample(ch))
                current_res *= 2
            else:
                self.upsamplers.append(nn.Identity())

        # ─────────── Output ───────────
        self.out_norm = nn.GroupNorm(min(32, ch), ch)
        self.out_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W)
            t: (B,) long tensor of timesteps

        Returns:
            (B, C_out, H, W) — 通常预测的是噪声 ε
        """
        # Time embedding
        t_emb = self.time_embedding(t)

        # Initial conv
        h = self.init_conv(x)
        skips = [h]

        # Encoder
        for blocks, attns, downsampler in zip(self.down_blocks, self.down_attns, self.downsamplers):
            for block, attn in zip(blocks, attns):
                h = block(h, t_emb)
                h = attn(h)
                skips.append(h)
            if not isinstance(downsampler, nn.Identity):
                h = downsampler(h)
                skips.append(h)

        # Mid
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Decoder
        for blocks, attns, upsampler in zip(self.up_blocks, self.up_attns, self.upsamplers):
            for block, attn in zip(blocks, attns):
                # 拼接对应 skip
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
                h = attn(h)
            if not isinstance(upsampler, nn.Identity):
                h = upsampler(h)

        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        return h


# ============================================================================
# 自测
# ============================================================================
if __name__ == '__main__':
    print("=== Testing unet.py ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试 ResBlock（依赖 TODO 8）
    try:
        block = ResBlock(64, 128, time_dim=512).to(device)
        x = torch.randn(2, 64, 16, 16, device=device)
        t_emb = torch.randn(2, 512, device=device)
        out = block(x, t_emb)
        assert out.shape == (2, 128, 16, 16), f"Expected (2,128,16,16), got {out.shape}"
        print("✅ ResBlock passed")
    except NotImplementedError:
        print("⚠️  ResBlock not implemented (TODO 8)")
        exit()

    # 测试完整 UNet（CIFAR-10 配置）
    try:
        model = UNet(in_channels=3, out_channels=3, base_channels=64,
                     channel_mult=(1, 2, 2), num_res_blocks=2,
                     attn_resolutions=(8,), image_size=32).to(device)
        x = torch.randn(2, 3, 32, 32, device=device)
        t = torch.tensor([100, 500], device=device)
        out = model(x, t)
        assert out.shape == x.shape
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  UNet param count: {n_params/1e6:.2f}M")
        print("✅ UNet (small) forward passed")
    except Exception as e:
        print(f"❌ UNet error: {e}")
        raise

    # 测试 MNIST 配置
    try:
        model_mnist = UNet(in_channels=1, out_channels=1, base_channels=32,
                           channel_mult=(1, 2), num_res_blocks=2,
                           attn_resolutions=(), image_size=28)
        # MNIST 28x28 不能简单下采样2次（28/4=7非偶），所以测试 32x32
        x = torch.randn(2, 1, 32, 32)
        t = torch.tensor([0, 999])
        out = model_mnist(x, t)
        assert out.shape == x.shape
        print("✅ UNet (MNIST-like) forward passed")
    except Exception as e:
        print(f"⚠️ UNet MNIST test: {e}")
