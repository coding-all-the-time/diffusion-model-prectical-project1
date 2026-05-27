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
    python train.py --config configs/mnist_cosine.yaml
重新启动：
    python train.py --config configs/mnist_cosine.yaml --resume runs/exp_mnist_cosine/ckpt/step_005000.pt
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
# from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt

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

        # EMA 模型本身不参与梯度计算，因此关闭所有参数的梯度以节省显存和计算量
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()# 更新过程不需要计算图
    def update(self, model: nn.Module):
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p.data, alpha=1 - self.decay)
            # 下划线结尾的方法（如 mul_ 和 add_）代表内联操作，即它们会直接修改当前张量内存里的值，而不需要开辟新的内存空间。
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
    with open(output_dir / 'config.yaml', 'w', encoding='utf-8') as f:
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
    scaler = torch.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16), device='cuda')

    # ─────────── 预加载 Checkpoint (提前获取 WandB Run ID) ───────────
    ckpt = None
    if cfg.get('resume'):
        print(f"[*] Loading checkpoint to resume: {cfg['resume']}")
        ckpt = torch.load(cfg['resume'], map_location=device, weights_only=False)

    # ─────────── WandB（可选） ───────────
    use_wandb = cfg.get('wandb', {}).get('enabled', False)
    if use_wandb:
        import wandb
        # 从 ckpt 中提取旧的 run_id
        run_id = ckpt.get('wandb_run_id') if ckpt else None
        wandb.init(
            project=cfg['wandb'].get('project', 'ddpm-course'),
            name=cfg['wandb'].get('run_name', None),
            config=cfg,
            id=run_id,         # 传入旧的 run_id（如果是新训练，这里是 None）
            resume="allow",    # 允许断点恢复合并图表
        )

    # ─────────── 训练循环 ───────────
    num_epochs = cfg['training']['num_epochs']
    log_every = cfg['training'].get('log_every', 50)
    sample_every = cfg['training'].get('sample_every', 1000)
    ckpt_every = cfg['training'].get('ckpt_every', 5000)
    grad_clip = cfg['training'].get('grad_clip', 1.0)

    # 设置最多保留的 checkpoint 数量，默认为 3
    max_keep_ckpts = cfg['training'].get('max_keep_ckpts', 3)

    # ─────────── 绘图记录初始化 ───────────
    use_plotting = cfg.get('plotting', {}).get('enabled', False)
    if use_plotting:
        record_every = cfg['plotting'].get('record_every', 50)
        plot_every = cfg['plotting'].get('plot_every', 1000)
        history_steps = []
        history_losses = []
        history_mems = []
        # 自动扫描并排序历史遗留的 Checkpoint
        ckpt_dir = output_dir / 'ckpt'
        saved_ckpts = sorted(list(ckpt_dir.glob('step_*.pt')))
        if len(saved_ckpts) > 0:
            print(f"[*] 已扫描到文件夹中存在的 {len(saved_ckpts)} 个历史 Checkpoint")

    global_step = 0
    start_epoch = 0
    
    # ─────────── 恢复断点状态 ───────────
    if ckpt:
        print(f"[*] Restoring model and optimizer states...")
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if ema is not None and 'ema' in ckpt:
            ema.load_state_dict(ckpt['ema'])
        if use_amp and amp_dtype == torch.float16 and 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
            
        global_step = ckpt.get('global_step', 0)
        start_epoch = ckpt.get('epoch', 0) + 1  # 从中断的下一个 epoch 继续
        
        # 修复学习率调度器（避免恢复后由于内部 step 归零导致学习率重新 warmup 到 0）
        if scheduler_lr is not None:
            scheduler_lr.last_epoch = global_step

    initial_step = global_step
    start_time = time.time()
    print(f"[train] Starting training for {num_epochs} epochs")

    for epoch in range(start_epoch, num_epochs):
        for batch_idx, x0 in enumerate(loader):
            x0 = x0.to(device, non_blocking=True)
            B = x0.shape[0]

            # 随机时间步
            t = torch.randint(0, schedule.T, (B,), device=device).long()

            # Forward + Loss
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(enabled=use_amp, dtype=amp_dtype, device_type='cuda'):
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

            # ─────────── 记录绘图数据 ───────────
            if use_plotting and global_step % record_every == 0:
                history_steps.append(global_step)
                history_losses.append(loss.item())
                if torch.cuda.is_available():
                    # mem_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
                    mem_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
                else:
                    mem_gb = 0.0
                history_mems.append(mem_gb)

            # ─────────── Logging ───────────
            if global_step % log_every == 0:
                elapsed = time.time() - start_time
                # 修改速度计算逻辑，只计算本次运行实际跑过的步数
                #     steps_per_sec = global_step / elapsed
                steps_per_sec = (global_step - initial_step) / elapsed 
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
                        # 新增：将自己获取的显存数据传给 WandB
                        'custom/gpu_mem_gb': torch.cuda.memory_reserved(device) / (1024 ** 3)
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

            # ─────────── 绘制并保存曲线 ───────────
            if use_plotting and global_step % plot_every == 0 and len(history_steps) > 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # 绘制Loss曲线
                ax1.plot(history_steps, history_losses, label='Loss', color='blue')
                ax1.set_title('Training Loss Curve')
                ax1.set_xlabel('Global Step')
                ax1.set_ylabel('Loss')
                ax1.grid(True)
                
                # 绘制显存曲线
                ax2.plot(history_steps, history_mems, label='VRAM (GB)', color='red')
                ax2.set_title('GPU Memory Usage Curve')
                ax2.set_xlabel('Global Step')
                ax2.set_ylabel('Memory (GB)')
                ax2.grid(True)
                
                plt.tight_layout()
                plot_path = output_dir / 'training_curves.png'
                plt.savefig(plot_path)
                plt.close()
                print(f"  [plot] curves saved to {plot_path}")

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
                # 将当前 WandB 的 run_id 保存进断点中
                if use_wandb and wandb.run is not None:
                    save_dict['wandb_run_id'] = wandb.run.id
                if ema is not None:
                    save_dict['ema'] = ema.state_dict()
                if use_amp and amp_dtype == torch.float16:
                    save_dict['scaler'] = scaler.state_dict()
                torch.save(save_dict, ckpt_path)
                print(f"  [ckpt] saved to {ckpt_path}")

                # 保留最近的 max_keep_ckpts 个 Checkpoint 的机制
                saved_ckpts.append(ckpt_path)
                if len(saved_ckpts) > max_keep_ckpts:
                    old_ckpt = saved_ckpts.pop(0) # 弹出列表第一个（最老的）
                    if old_ckpt.exists():
                        try:
                            old_ckpt.unlink() # 从硬盘上删除文件
                            print(f"  [ckpt] removed old checkpoint: {old_ckpt}")
                        except Exception as e:
                            print(f"  [ckpt] warning: failed to remove {old_ckpt}: {e}")

    # ─────────── 训练结束 ───────────
    final_ckpt = output_dir / 'ckpt' / 'final.pt'
    save_dict = {
        'global_step': global_step,
        'model': model.state_dict(),
        'config': cfg,
    }
    if use_wandb and wandb.run is not None:
        save_dict['wandb_run_id'] = wandb.run.id
    if ema is not None:
        save_dict['ema'] = ema.state_dict()
    torch.save(save_dict, final_ckpt)
    print(f"\n[done] Final checkpoint saved to {final_ckpt}")
    print(f"[done] Total training time: {(time.time() - start_time)/60:.1f} min")

    # 打印显存峰值统计 (除以 1024^3 将 Byte 转换为 GB)
    if torch.cuda.is_available():
        # peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        peak_memory = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        print(f"[done] Peak GPU memory usage: {peak_memory:.2f} GB")

    # ─────────── 训练结束时保存最终曲线（新增） ───────────
    if use_plotting and len(history_steps) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(history_steps, history_losses, label='Loss', color='blue')
        ax1.set_title('Final Training Loss Curve')
        ax1.set_xlabel('Global Step')
        ax1.set_ylabel('Loss')
        ax1.grid(True)

        ax2.plot(history_steps, history_mems, label='VRAM (GB)', color='red')
        ax2.set_title('Final GPU Memory Usage Curve')
        ax2.set_xlabel('Global Step')
        ax2.set_ylabel('Memory (GB)')
        ax2.grid(True)

        plt.tight_layout()
        plot_path = output_dir / 'training_curves_final.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"[plot] Final curves saved to {plot_path}")
    
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
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    if args.output_dir:
        cfg['output_dir'] = args.output_dir
    if args.seed is not None:
        cfg['seed'] = args.seed
    if args.resume is not None:
        cfg['resume'] = args.resume

    print("=" * 60)
    print("DDPM Training")
    print("=" * 60)
    print(yaml.dump(cfg, default_flow_style=False))
    print("=" * 60)

    train(cfg)


if __name__ == '__main__':
    main()
