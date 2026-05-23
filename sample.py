"""
DDPM 采样脚本

加载训练好的 checkpoint，生成图像并保存。

用法：
    # 生成 64 张图，使用 EMA 权重
    python sample.py --ckpt runs/exp_mnist/ckpt/final.pt --num_samples 64

    # 不使用 EMA（对比用）
    python sample.py --ckpt runs/exp_mnist/ckpt/final.pt --no_ema
"""

import argparse
from pathlib import Path
import torch
import yaml
import torchvision.utils as vutils

from schedule import DDPMSchedule
from diffusion import p_sample_loop
from model import UNet
from dataset import denormalize


def load_model_from_ckpt(ckpt_path: str, use_ema: bool = True):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg = ckpt['config']

    model = UNet(**cfg['model'])
    schedule = DDPMSchedule(**cfg['diffusion'])

    if use_ema and 'ema' in ckpt:
        model.load_state_dict(ckpt['ema'])
        print(f"[load] Using EMA weights from {ckpt_path}")
    else:
        model.load_state_dict(ckpt['model'])
        print(f"[load] Using model weights from {ckpt_path}")

    return model, schedule, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--no_ema', action='store_true', help='Do not use EMA weights')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_grid', action='store_true', help='Save as grid image')
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model, schedule, cfg = load_model_from_ckpt(args.ckpt, use_ema=not args.no_ema)
    model = model.to(device).eval()
    schedule = schedule.to(device)

    # 输出目录
    if args.output_dir is None:
        ckpt_dir = Path(args.ckpt).parent
        suffix = '_ema' if not args.no_ema else '_noema'
        out_dir = ckpt_dir.parent / f'samples_inference{suffix}'
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] {out_dir}")

    # 分批采样
    image_size = cfg['model']['image_size']
    in_ch = cfg['model']['in_channels']

    all_samples = []
    n_remaining = args.num_samples
    batch_idx = 0

    while n_remaining > 0:
        bs = min(args.batch_size, n_remaining)
        print(f"[sample] batch {batch_idx}: generating {bs} samples...")
        samples = p_sample_loop(
            model,
            shape=(bs, in_ch, image_size, image_size),
            schedule=schedule,
            device=device,
        )
        samples = denormalize(samples).cpu()

        # 单张保存
        for i in range(bs):
            idx = batch_idx * args.batch_size + i
            vutils.save_image(samples[i], out_dir / f'sample_{idx:04d}.png')

        all_samples.append(samples)
        n_remaining -= bs
        batch_idx += 1

    # 保存网格
    if args.save_grid:
        all_samples = torch.cat(all_samples, dim=0)
        nrow = int(args.num_samples ** 0.5)
        grid = vutils.make_grid(all_samples, nrow=nrow, padding=2)
        grid_path = out_dir / 'grid.png'
        vutils.save_image(grid, grid_path)
        print(f"[grid] saved to {grid_path}")

    print(f"[done] Generated {args.num_samples} samples in {out_dir}")


if __name__ == '__main__':
    main()
