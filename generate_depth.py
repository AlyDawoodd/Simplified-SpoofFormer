"""
Regenerate depth maps for all CASIA-FASD images using MiDaS.

Replaces the pre-computed depth maps (which are black for spoof images)
with proper MiDaS-estimated depth maps for ALL images.

Usage:
    python generate_depth.py --data_root data/casiafasd/split
    
Output:
    Overwrites depth/ folders in train/, val/, test/ with MiDaS depth maps.
    Original pre-computed depth maps are backed up to depth_original/.
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def load_midas(device):
    """Load MiDaS_small model."""
    print("[MiDaS] Loading MiDaS_small...")
    model = torch.hub.load(
        "intel-isl/MiDaS", "MiDaS_small",
        trust_repo=True, verbose=False
    )
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    print("[MiDaS] Loaded successfully")
    return model


@torch.no_grad()
def estimate_depth(model, img_bgr: np.ndarray, device, size: int = 256) -> np.ndarray:
    """
    Estimate depth from BGR image using MiDaS.
    Returns uint8 grayscale depth map same size as input.
    """
    h, w = img_bgr.shape[:2]

    # BGR -> RGB -> float tensor
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor  = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    # Resize to MiDaS input size (must be divisible by 32)
    tensor_resized = F.interpolate(tensor, size=(size, size),
                                   mode='bilinear', align_corners=False)

    # Estimate depth
    depth = model(tensor_resized)              # (1, size, size)
    depth = depth.squeeze().cpu().numpy()      # (size, size)

    # Resize back to original size
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize to uint8 [0, 255]
    depth = depth - depth.min()
    dmax  = depth.max()
    if dmax > 0:
        depth = depth / dmax
    depth = (depth * 255).astype(np.uint8)

    return depth


def process_split(model, split_dir: Path, device, backup: bool = True):
    """Process all images in a split folder."""
    color_dir = split_dir / 'color'
    depth_dir = split_dir / 'depth'
    backup_dir = split_dir / 'depth_original'

    if not color_dir.exists():
        print(f"  Skipping {split_dir.name} — color dir not found")
        return

    # Backup original depth maps
    if backup and depth_dir.exists() and not backup_dir.exists():
        print(f"  Backing up original depth maps to {backup_dir.name}/")
        shutil.copytree(depth_dir, backup_dir)

    depth_dir.mkdir(exist_ok=True)

    # Get all color images
    images = sorted(list(color_dir.glob('*.jpg')) +
                    list(color_dir.glob('*.png')))

    print(f"  Processing {len(images)} images in {split_dir.name}/...")

    ok, failed = 0, 0
    for img_path in tqdm(images, desc=f"  {split_dir.name}"):
        img = cv2.imread(str(img_path))
        if img is None:
            failed += 1
            continue

        depth = estimate_depth(model, img, device)
        out_path = depth_dir / img_path.name
        cv2.imwrite(str(out_path), depth)
        ok += 1

    print(f"  Done — {ok} generated, {failed} failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/casiafasd/split',
                        help='Path to split folder with train/val/test subfolders')
    parser.add_argument('--size',      default=256, type=int,
                        help='MiDaS inference size (must be divisible by 32)')
    parser.add_argument('--no_backup', action='store_true',
                        help='Skip backing up original depth maps')
    args = parser.parse_args()

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root     = Path(args.data_root)
    print(f"\n[Config] data_root={root} | device={device} | midas_size={args.size}")

    # Load MiDaS once
    model = load_midas(device)

    # Process each split
    for split in ['train', 'val', 'test']:
        split_dir = root / split
        if split_dir.exists():
            print(f"\n[{split.upper()}]")
            process_split(model, split_dir, device,
                          backup=not args.no_backup)

    print(f"\n[Done] MiDaS depth maps generated for all splits.")
    print(f"Original depth maps backed up to depth_original/ in each split.")
    print(f"\nNow run training:")
    print(f"  python train.py --data_root data/casiafasd/split \\")
    print(f"    --model_size small --epochs 50 --lr 5e-5 \\")
    print(f"    --no_midas --num_workers 0 --batch_size 4 --img_size 128")


if __name__ == '__main__':
    main()
