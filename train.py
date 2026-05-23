"""
DDPM 训练脚本

完整训练流程，包含：
- 混合精度（fp16/bf16）
- EMA（指数移动平均）
- 梯度裁剪
- WandB 日志
- 周期性采样可视化
- Checkpoint 保存与恢复

用法：
    python train.py --config configs/mnist.yaml

学生不需要修改本文件即可运行，但建议读懂训练循环结构。
"""

import argparse
import copy
import os
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from schedule import DDPMSchedule
from diffusion import p_losses, p_sample_loop
from model import UNet
from dataset import get_dataloader, denormalize


# ============================================================================
# EMA helper
# ============================================================================
class EMA:
    """
    Exponential Moving Average for model parameters.

    用法：
        ema = EMA(model, decay=0.9999)
        # 训练循环中：
        loss.backward(); optimizer.step()
        ema.update(model)
        # 采样时：
        ema_model = ema.copy_to(model)
        samples = sample(ema_model, ...)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        # 用 deepcopy 而非 state_dict，确保 buffer 也被复制
        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p.data, alpha=1 - self.decay)
        # buffer（如 batchnorm running stats）直接复制
        for ema_b, b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, sd):
        self.ema_model.load_state_dict(sd)


# ============================================================================
# 主训练函数
# ============================================================================
def train(cfg: Dict[str, Any]):
    # ─────────── 准备工作 ───────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'samples').mkdir(exist_ok=True)
    (output_dir / 'ckpt').mkdir(exist_ok=True)

    # 随机种子
    seed = cfg.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 保存配置
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f)

    # ─────────── 数据 ───────────
    loader = get_dataloader(
        name=cfg['dataset']['name'],
        batch_size=cfg['dataset']['batch_size'],
        root=cfg['dataset'].get('root', './data'),
        image_size=cfg['dataset'].get('image_size', None),
        num_workers=cfg['dataset'].get('num_workers', 4),
    )
    print(f"[data] {cfg['dataset']['name']}: {len(loader.dataset)} samples")

    # ─────────── 模型 ───────────
    model = UNet(**cfg['model']).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] UNet param count: {n_params/1e6:.2f}M")

    # ─────────── Schedule ───────────
    schedule = DDPMSchedule(**cfg['diffusion']).to(device)
    print(f"[schedule] T={schedule.T}, type={cfg['diffusion'].get('beta_schedule', 'linear')}")

    # ─────────── Optimizer ───────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['optimizer']['lr'],
        weight_decay=cfg['optimizer'].get('weight_decay', 0.0),
        betas=(0.9, 0.999),
    )

    # Warmup
    warmup_steps = cfg['optimizer'].get('warmup_steps', 0)
    if warmup_steps > 0:
        def lr_lambda(step):
            return min(step / warmup_steps, 1.0)
        scheduler_lr = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler_lr = None

    # ─────────── EMA ───────────
    use_ema = cfg.get('ema_decay', 0) > 0
    ema = EMA(model, decay=cfg.get('ema_decay', 0.9999)) if use_ema else None

    # ─────────── 混合精度 ───────────
    use_amp = cfg.get('mixed_precision', 'no') in ['fp16', 'bf16']
    amp_dtype = torch.bfloat16 if cfg.get('mixed_precision') == 'bf16' else torch.float16
    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    # ─────────── WandB（可选） ───────────
    use_wandb = cfg.get('wandb', {}).get('enabled', False)
    if use_wandb:
        import wandb
        wandb.init(
            project=cfg['wandb'].get('project', 'ddpm-course'),
            name=cfg['wandb'].get('run_name', None),
            config=cfg,
        )

    # ─────────── 训练循环 ───────────
    num_epochs = cfg['training']['num_epochs']
    log_every = cfg['training'].get('log_every', 50)
    sample_every = cfg['training'].get('sample_every', 1000)
    ckpt_every = cfg['training'].get('ckpt_every', 5000)
    grad_clip = cfg['training'].get('grad_clip', 1.0)

    global_step = 0
    start_time = time.time()
    print(f"[train] Starting training for {num_epochs} epochs")

    for epoch in range(num_epochs):
        for batch_idx, x0 in enumerate(loader):
            x0 = x0.to(device, non_blocking=True)
            B = x0.shape[0]

            # 随机时间步
            t = torch.randint(0, schedule.T, (B,), device=device).long()

            # Forward + Loss
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp, dtype=amp_dtype):
                loss = p_losses(model, x0, t, schedule)

            # Backward
            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            if scheduler_lr is not None:
                scheduler_lr.step()

            # EMA update
            if ema is not None:
                ema.update(model)

            global_step += 1

            # ─────────── Logging ───────────
            if global_step % log_every == 0:
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed
                lr_now = optimizer.param_groups[0]['lr']
                print(f"[ep {epoch:03d} step {global_step:06d}] "
                      f"loss={loss.item():.4f} lr={lr_now:.2e} "
                      f"({steps_per_sec:.1f} step/s)")
                if use_wandb:
                    wandb.log({
                        'loss': loss.item(),
                        'lr': lr_now,
                        'epoch': epoch,
                        'steps_per_sec': steps_per_sec,
                    }, step=global_step)

            # ─────────── Sampling ───────────
            if global_step % sample_every == 0:
                sample_model = ema.ema_model if ema is not None else model
                sample_model.eval()
                with torch.no_grad():
                    samples = p_sample_loop(
                        sample_model,
                        shape=(16, cfg['model']['in_channels'],
                               cfg['model']['image_size'], cfg['model']['image_size']),
                        schedule=schedule,
                        device=device,
                    )
                samples = denormalize(samples)
                grid = vutils.make_grid(samples, nrow=4, padding=2)
                save_path = output_dir / 'samples' / f'step_{global_step:06d}.png'
                vutils.save_image(grid, save_path)
                print(f"  [sample] saved to {save_path}")
                if use_wandb:
                    wandb.log({'samples': wandb.Image(grid)}, step=global_step)
                sample_model.train()

            # ─────────── Checkpoint ───────────
            if global_step % ckpt_every == 0:
                ckpt_path = output_dir / 'ckpt' / f'step_{global_step:06d}.pt'
                save_dict = {
                    'global_step': global_step,
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': cfg,
                }
                if ema is not None:
                    save_dict['ema'] = ema.state_dict()
                if use_amp and amp_dtype == torch.float16:
                    save_dict['scaler'] = scaler.state_dict()
                torch.save(save_dict, ckpt_path)
                print(f"  [ckpt] saved to {ckpt_path}")

    # ─────────── 训练结束 ───────────
    final_ckpt = output_dir / 'ckpt' / 'final.pt'
    save_dict = {
        'global_step': global_step,
        'model': model.state_dict(),
        'config': cfg,
    }
    if ema is not None:
        save_dict['ema'] = ema.state_dict()
    torch.save(save_dict, final_ckpt)
    print(f"\n[done] Final checkpoint saved to {final_ckpt}")
    print(f"[done] Total training time: {(time.time() - start_time)/60:.1f} min")

    if use_wandb:
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='YAML config file')
    parser.add_argument('--output_dir', type=str, default=None, help='覆盖配置中的 output_dir')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None, help='checkpoint to resume from')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    if args.output_dir:
        cfg['output_dir'] = args.output_dir
    if args.seed is not None:
        cfg['seed'] = args.seed

    print("=" * 60)
    print("DDPM Training")
    print("=" * 60)
    print(yaml.dump(cfg, default_flow_style=False))
    print("=" * 60)

    train(cfg)


if __name__ == '__main__':
    main()
