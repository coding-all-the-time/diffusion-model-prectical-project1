"""
DDPM 评估脚本：FID 计算

使用 torchmetrics 的 FrechetInceptionDistance 计算生成样本与真实样本的 FID。

依赖：
    pip install torchmetrics[image] torch-fidelity

用法：
    # 评估 checkpoint 上的 FID（生成 5000 张样本对比真实数据）
    python evaluate.py --ckpt runs/exp_cifar10/ckpt/final.pt --num_samples 5000 --batch_size 64
"""

import argparse
from pathlib import Path
import torch
import yaml

from schedule import DDPMSchedule
from diffusion import p_sample_loop
from model import UNet
from dataset import get_dataset, denormalize


def compute_fid(
    model,
    schedule,
    real_dataset,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    image_size: int,
    in_channels: int,
) -> float:
    """
    计算 FID（Frechet Inception Distance）。

    FID 衡量生成分布与真实分布在 Inception 特征空间上的距离，越低越好。
    建议至少 5000 样本以获得稳定值；CIFAR-10 论文标准是 50000。

    Returns:
        FID score (float)
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        raise ImportError(
            "需要安装 torchmetrics[image]:\n"
            "  pip install torchmetrics[image] torch-fidelity"
        )

    # FID 要求输入是 uint8 的 [0, 255] RGB 图（即使是灰度也要扩展到 3 通道）
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    fid.reset()

    # ─────────── 1. 喂入真实样本 ───────────
    print(f"[FID] Loading {num_samples} real samples...")
    real_loader = torch.utils.data.DataLoader(
        real_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, drop_last=False,
    )

    n_real_loaded = 0
    for batch in real_loader:
        if n_real_loaded >= num_samples:
            break
        x = batch.to(device)
        # [-1,1] → [0, 255] uint8
        x = denormalize(x)
        if in_channels == 1:
            x = x.repeat(1, 3, 1, 1)
        x_uint8 = (x * 255).clamp(0, 255).to(torch.uint8)
        fid.update(x_uint8, real=True)
        n_real_loaded += x.shape[0]
    print(f"[FID] Real samples loaded: {n_real_loaded}")

    # ─────────── 2. 生成样本 ───────────
    print(f"[FID] Generating {num_samples} fake samples...")
    n_generated = 0
    while n_generated < num_samples:
        bs = min(batch_size, num_samples - n_generated)
        with torch.no_grad():
            samples = p_sample_loop(
                model,
                shape=(bs, in_channels, image_size, image_size),
                schedule=schedule,
                device=device,
            )
        samples = denormalize(samples)
        if in_channels == 1:
            samples = samples.repeat(1, 3, 1, 1)
        samples_uint8 = (samples * 255).clamp(0, 255).to(torch.uint8)
        fid.update(samples_uint8, real=False)
        n_generated += bs
        if n_generated % (batch_size * 10) == 0 or n_generated >= num_samples:
            print(f"  generated {n_generated}/{num_samples}")

    # ─────────── 3. 计算 FID ───────────
    score = fid.compute().item()
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=5000,
                        help='Samples for FID (recommend >=5000)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--no_ema', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_root', type=str, default='./data')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载 checkpoint
    ckpt = torch.load(args.ckpt, map_location='cpu')
    cfg = ckpt['config']

    model = UNet(**cfg['model']).to(device)
    schedule = DDPMSchedule(**cfg['diffusion']).to(device)

    if not args.no_ema and 'ema' in ckpt:
        model.load_state_dict(ckpt['ema'])
        weight_type = 'EMA'
    else:
        model.load_state_dict(ckpt['model'])
        weight_type = 'raw'
    model.eval()
    print(f"[model] Loaded {weight_type} weights from {args.ckpt}")

    # 真实数据集
    image_size = cfg['model']['image_size']
    in_channels = cfg['model']['in_channels']
    real_dataset = get_dataset(
        cfg['dataset']['name'],
        root=args.data_root,
        image_size=image_size,
        train=True,
    )

    # 计算 FID
    fid_score = compute_fid(
        model=model,
        schedule=schedule,
        real_dataset=real_dataset,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=device,
        image_size=image_size,
        in_channels=in_channels,
    )

    print("\n" + "=" * 50)
    print(f"FID @ {args.num_samples} samples: {fid_score:.4f}")
    print(f"Weights: {weight_type}")
    print(f"Checkpoint: {args.ckpt}")
    print("=" * 50)

    # 保存到文件（便于实验日志）
    result_path = Path(args.ckpt).parent / f'fid_{args.num_samples}_{weight_type}.txt'
    with open(result_path, 'w') as f:
        f.write(f"FID: {fid_score:.4f}\n")
        f.write(f"num_samples: {args.num_samples}\n")
        f.write(f"weights: {weight_type}\n")
        f.write(f"ckpt: {args.ckpt}\n")
        f.write(f"seed: {args.seed}\n")
    print(f"[saved] {result_path}")


if __name__ == '__main__':
    main()
