"""
Training script for Dual-Stream SpoofFormer on CASIA-FASD.

Usage:
    python train.py --data_root data/casiafasd --epochs 30
    python train.py --data_root data/casiafasd --epochs 30 --model_size small  # faster
    python train.py --data_root data/casiafasd --epochs 30 --no_midas          # use pre-computed depth

Outputs saved to: checkpoints/
    best_model.pth    <- best validation ACER
    last_model.pth    <- final epoch
    training_log.csv
"""

import os
import csv
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from models.model import spoofformer_small, spoofformer_base, spoofformer_large
from data.dataset import build_dataloaders
from utils.metrics import evaluate_all


# ─────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Dual-Stream SpoofFormer")
    p.add_argument('--data_root',    default='data/casiafasd')
    p.add_argument('--model_size',   default='base',
                   choices=['small', 'base', 'large'])
    p.add_argument('--epochs',       default=30,   type=int)
    p.add_argument('--batch_size',   default=16,   type=int)
    p.add_argument('--lr',           default=1e-4, type=float)
    p.add_argument('--weight_decay', default=1e-4, type=float)
    p.add_argument('--warmup_epochs',default=5,    type=int)
    p.add_argument('--img_size',     default=224,  type=int)
    p.add_argument('--num_workers',  default=0,    type=int,
                   help='0 recommended on Windows to avoid multiprocessing issues')
    p.add_argument('--val_split',    default=0.2,  type=float,
                   help='Fraction of train set to use as validation (0=use test split)')
    p.add_argument('--no_midas',     action='store_true',
                   help='Use pre-computed depth maps instead of MiDaS (recommended for CASIA-FASD)')
    p.add_argument('--checkpoint_dir', default='checkpoints')
    p.add_argument('--aug_level', default='none', choices=['none','light','full'],
                   help='Augmentation level: none=resize only, light=flip+rotation, full=all')
    p.add_argument('--no_oversample', action='store_true',
                   help='Disable oversampling of minority class')
    p.add_argument('--dropout',       default=0.3,  type=float,
                   help='Dropout rate — higher helps on small datasets')
    p.add_argument('--early_stop',   default=7, type=int,
                   help='Stop if val ACER does not improve for this many epochs (0=disabled)')
    p.add_argument('--resume',       default=None)
    return p.parse_args()


# ─────────────────────────────────────────────
# LR SCHEDULER: Warmup + Cosine
# ─────────────────────────────────────────────

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup    = warmup_epochs
        self.total     = total_epochs
        self.min_lr    = min_lr
        self.base_lrs  = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup:
            factor = (epoch + 1) / self.warmup
        else:
            progress = (epoch - self.warmup) / max(1, self.total - self.warmup)
            factor   = 0.5 * (1 + np.cos(np.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.min_lr, base_lr * factor)


# ─────────────────────────────────────────────
# TRAIN ONE EPOCH
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch_idx, (rgb, depth, labels) in enumerate(loader):
        rgb    = rgb.to(device, non_blocking=True)
        depth  = depth.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast('cuda', enabled=(scaler is not None)):
            logits = model(rgb, depth)
            loss   = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        if batch_idx % 20 == 0:
            print(f"  Epoch {epoch} [{batch_idx}/{len(loader)}] "
                  f"loss={loss.item():.4f} "
                  f"acc={correct/total*100:.1f}%")

    return total_loss / len(loader), correct / total


# ─────────────────────────────────────────────
# VALIDATE
# ─────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_scores = []

    for rgb, depth, labels in loader:
        rgb    = rgb.to(device, non_blocking=True)
        depth  = depth.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(rgb, depth)
        loss   = criterion(logits, labels)
        total_loss += loss.item()

        probs  = torch.softmax(logits, dim=1)[:, 1]   # P(spoof)
        all_labels.append(labels.cpu().numpy())
        all_scores.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)
    metrics    = evaluate_all(all_labels, all_scores, verbose=False)
    return total_loss / len(loader), metrics


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  Dual-Stream SpoofFormer Training")
    print(f"  device={device} | model={args.model_size} | epochs={args.epochs}")
    print(f"  data={args.data_root} | batch={args.batch_size}")
    print(f"{'='*60}\n")

    # ── Dataloaders ───────────────────────────
    train_loader, val_loader = build_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        oversample=not args.no_oversample,
        aug_level=args.aug_level,
    )

    # ── Model ─────────────────────────────────
    # use_midas=False because CASIA-FASD already has pre-computed depth maps
    builders = {'small': spoofformer_small,
                'base':  spoofformer_base,
                'large': spoofformer_large}
    model = builders[args.model_size](
        img_size=args.img_size,
        use_midas=not args.no_midas,
        dropout=args.dropout,
    ).to(device)
    print(f"[Model] Trainable parameters: {model.get_num_params():,}")
    print(f"[Model] Depth source: {'pre-computed (dataset)' if args.no_midas else 'MiDaS'}\n")

    # ── Loss with class weights (handles real/spoof imbalance) ──
    # CASIA-FASD: ~24% real, ~76% spoof — upweight the minority real class
    try:
        dataset = train_loader.dataset
        samples = dataset.dataset.samples if hasattr(dataset, "dataset") else dataset.samples
        indices = dataset.indices if hasattr(dataset, "indices") else range(len(samples))
        labels_list = [samples[i][2] for i in indices]
        n_real  = sum(1 for l in labels_list if l == 0)
        n_spoof = sum(1 for l in labels_list if l == 1)
        n_total = n_real + n_spoof
        w_real  = n_total / (2 * max(n_real,  1))
        w_spoof = n_total / (2 * max(n_spoof, 1))
    except Exception:
        w_real, w_spoof = 2.0, 0.66
    class_weights = torch.tensor([w_real, w_spoof], dtype=torch.float).to(device)
    print(f"[Loss] Class weights — real: {w_real:.2f}  spoof: {w_spoof:.2f}")
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # ── Optimizer ─────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # ── LR Scheduler ──────────────────────────
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
    )

    # ── AMP ───────────────────────────────────
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    if scaler:
        print("[AMP] Mixed precision (FP16) enabled\n")

    # ── Resume ────────────────────────────────
    start_epoch       = 0
    best_acer         = float('inf')
    epochs_no_improve = 0
    if args.resume and Path(args.resume).exists():
        ckpt        = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_acer   = ckpt.get('best_acer', float('inf'))
        print(f"[Resume] Loaded from epoch {ckpt['epoch']}\n")

    # ── Checkpoints + CSV ─────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / 'training_log.csv'

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc',
                         'val_loss', 'auc', 'eer',
                         'apcer', 'bpcer', 'acer', 'lr'])

    # ── Training Loop ─────────────────────────
    for epoch in range(start_epoch, args.epochs):
        scheduler.step(epoch)
        lr = optimizer.param_groups[0]['lr']

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch)
        val_loss, metrics = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"\nEpoch {epoch:03d}/{args.epochs-1} | "
              f"lr={lr:.2e} | time={elapsed:.1f}s")
        print(f"  Train : loss={train_loss:.4f}  acc={train_acc*100:.2f}%")
        print(f"  Val   : loss={val_loss:.4f}  "
              f"AUC={metrics['auc']*100:.2f}%  "
              f"ACER={metrics['acer']*100:.2f}%  "
              f"EER={metrics['eer']*100:.2f}%")

        # CSV log
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, f"{train_loss:.4f}", f"{train_acc:.4f}",
                f"{val_loss:.4f}",
                f"{metrics['auc']:.4f}",   f"{metrics['eer']:.4f}",
                f"{metrics['apcer']:.4f}", f"{metrics['bpcer']:.4f}",
                f"{metrics['acer']:.4f}",  f"{lr:.2e}",
            ])

        # Save last
        torch.save({
            'epoch':     epoch,
            'model':     model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acer': best_acer,
            'metrics':   metrics,
            'args':      vars(args),
        }, ckpt_dir / 'last_model.pth')

        # Save best
        if metrics['acer'] < best_acer:
            best_acer = metrics['acer']
            torch.save({
                'epoch':     epoch,
                'model':     model.state_dict(),
                'best_acer': best_acer,
                'metrics':   metrics,
                'args':      vars(args),
            }, ckpt_dir / 'best_model.pth')
            print(f"  ✓ New best ACER: {best_acer*100:.2f}% — saved")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if args.early_stop > 0 and epochs_no_improve >= args.early_stop:
                print(f"\n[Early Stop] No improvement for {args.early_stop} epochs. "
                      f"Best ACER={best_acer*100:.2f}% at saved checkpoint.")
                break

    print(f"\n[Done] Best ACER : {best_acer*100:.2f}%")
    print(f"[Done] Log       : {log_path}")
    print(f"[Done] Checkpoint: {ckpt_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()
