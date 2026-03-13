"""
Evaluation script for Dual-Stream SpoofFormer.

Computes: AUC, EER, APCER, BPCER, ACER
Outputs:  ROC curve plot + metrics.json

Usage:
    python evaluate.py --data_root data/casiafasd/split \
                       --checkpoint checkpoints/best_model.pth \
                       --no_midas
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from data.dataset import CASIAFASDDataset
from models.model import spoofformer_small, spoofformer_base, spoofformer_large
from utils.metrics import evaluate_all


# ─────────────────────────────────────────────
# COLLECT PREDICTIONS
# ─────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_labels, all_scores = [], []

    for rgb, depth, labels in loader:
        rgb    = rgb.to(device)
        depth  = depth.to(device)
        logits = model(rgb, depth)
        probs  = torch.softmax(logits, dim=1)[:, 1]   # P(spoof)
        all_labels.append(labels.numpy())
        all_scores.append(probs.cpu().numpy())

    return np.concatenate(all_labels), np.concatenate(all_scores)


# ─────────────────────────────────────────────
# ROC CURVE PLOT
# ─────────────────────────────────────────────

def plot_roc_curve(y_true, y_pred, auc, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC Curve (AUC = {auc*100:.2f}%)')
    plt.plot([0, 1], [0, 1], color='navy', lw=1,
             linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (APCER direction)')
    plt.ylabel('True Positive Rate')
    plt.title('Dual-Stream SpoofFormer — ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] ROC curve saved: {save_path}")


# ─────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Dual-Stream SpoofFormer")
    p.add_argument('--data_root',    default='data/casiafasd/split')
    p.add_argument('--checkpoint',   default='checkpoints/best_model.pth')
    p.add_argument('--model_size',   default='small',
                   choices=['small', 'base', 'large'])
    p.add_argument('--split',        default='test',
                   choices=['train', 'val', 'test'],
                   help='Which split to evaluate on (default: test)')
    p.add_argument('--batch_size',   default=8,   type=int)
    p.add_argument('--img_size',     default=128, type=int)
    p.add_argument('--num_workers',  default=0,   type=int)
    p.add_argument('--no_midas',     action='store_true')
    p.add_argument('--output_dir',   default='evaluation_results')
    p.add_argument('--threshold',    default=None, type=float)
    return p.parse_args()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Dual-Stream SpoofFormer Evaluation")
    print(f"  split={args.split} | device={device}")
    print(f"  checkpoint={args.checkpoint}")
    print(f"{'='*55}\n")

    # ── Dataset ───────────────────────────────
    dataset = CASIAFASDDataset(
        root=args.data_root,
        split=args.split,
        img_size=args.img_size,
        oversample_real=False,
        aug_level='none',
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"[Dataset] {len(dataset)} samples for evaluation\n")

    # ── Model ─────────────────────────────────
    builders = {'small': spoofformer_small,
                'base':  spoofformer_base,
                'large': spoofformer_large}
    model = builders[args.model_size](
        img_size=args.img_size,
        use_midas=not args.no_midas,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    best_epoch = ckpt.get('epoch', '?')
    print(f"[Model] Loaded from epoch {best_epoch}")
    print(f"[Model] Val ACER at save: {ckpt.get('best_acer', 0)*100:.2f}%\n")

    # ── Predictions ───────────────────────────
    y_true, y_pred = collect_predictions(model, loader, device)

    # ── Metrics ───────────────────────────────
    print("[Metrics]")
    metrics = evaluate_all(y_true, y_pred,
                           threshold=args.threshold, verbose=True)

    # ── ROC Curve ─────────────────────────────
    plot_roc_curve(y_true, y_pred, metrics['auc'],
                   save_path=str(out_dir / 'roc_curve.png'))

    # ── Save JSON ─────────────────────────────
    report = {
        'checkpoint':    args.checkpoint,
        'best_epoch':    best_epoch,
        'eval_split':    args.split,
        'n_samples':     int(len(y_true)),
        'n_real':        int((y_true == 0).sum()),
        'n_spoof':       int((y_true == 1).sum()),
        'metrics': {
            'AUC':   f"{metrics['auc']*100:.2f}%",
            'EER':   f"{metrics['eer']*100:.2f}%",
            'APCER': f"{metrics['apcer']*100:.2f}%",
            'BPCER': f"{metrics['bpcer']*100:.2f}%",
            'ACER':  f"{metrics['acer']*100:.2f}%",
        },
        'threshold': float(metrics['threshold']),
        'comparison': {
            'paper_AUC':        '99.22%',
            'paper_ACER':       '0.81%',
            'our_AUC':          f"{metrics['auc']*100:.2f}%",
            'our_ACER':         f"{metrics['acer']*100:.2f}%",
            'expected_range':   'AUC 85-97%, ACER 3-15% for simplified model',
        }
    }

    json_path = out_dir / 'metrics.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Final Results on {args.split.upper()} split")
    print(f"  AUC   = {metrics['auc']*100:.2f}%")
    print(f"  EER   = {metrics['eer']*100:.2f}%")
    print(f"  APCER = {metrics['apcer']*100:.2f}%")
    print(f"  BPCER = {metrics['bpcer']*100:.2f}%")
    print(f"  ACER  = {metrics['acer']*100:.2f}%")
    print(f"{'='*55}")
    print(f"\n[Saved] {json_path}")
    print(f"[Saved] {out_dir}/roc_curve.png")


if __name__ == '__main__':
    main()
