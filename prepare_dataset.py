"""
Creates a proper subject-level 70/20/10 train/val/test split for CASIA-FASD.
Prevents data leakage — no subject appears in more than one split.

Usage: python prepare_dataset.py --data_root data/casiafasd
Output:
    data/casiafasd/split/
        train/  color/ + depth/   (70% of subjects)
        val/    color/ + depth/   (20% of subjects)
        test/   color/ + depth/   (10% of subjects)
"""

import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict


def parse_subject_id(filename: str) -> str:
    """'10_1.avi_100_real.jpg' -> '10'"""
    return filename.split('_')[0]


def copy_sample(src: Path, dst_color: Path, dst_depth: Path,
                depth_dirs: list):
    """Copy color image and matching depth map to destination."""
    shutil.copy2(src, dst_color / src.name)
    for depth_dir in depth_dirs:
        depth_src = depth_dir / src.name
        if depth_src.exists():
            shutil.copy2(depth_src, dst_depth / src.name)
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',   default='data/casiafasd')
    parser.add_argument('--train_ratio', default=0.7,  type=float)
    parser.add_argument('--val_ratio',   default=0.2,  type=float)
    parser.add_argument('--seed',        default=42,   type=int)
    args = parser.parse_args()

    assert abs(args.train_ratio + args.val_ratio - 0.9) < 1e-6 or \
           args.train_ratio + args.val_ratio < 1.0, \
           "train + val must be < 1.0 (remaining goes to test)"

    root = Path(args.data_root)
    random.seed(args.seed)

    # ── Collect all unique color images ──────────────────
    all_color = []
    depth_dirs = []
    for folder in ['train_img/train_img', 'test_img/test_img']:
        color_dir = root / folder / 'color'
        depth_dir = root / folder / 'depth'
        if color_dir.exists():
            all_color.extend(color_dir.glob('*.jpg'))
        if depth_dir.exists():
            depth_dirs.append(depth_dir)

    # Deduplicate by filename
    seen, unique = set(), []
    for p in all_color:
        if p.name not in seen:
            seen.add(p.name)
            unique.append(p)

    print(f"Total unique images : {len(unique)}")

    # ── Group by subject ──────────────────────────────────
    subjects = defaultdict(list)
    for p in unique:
        subjects[parse_subject_id(p.name)].append(p)

    subject_ids = sorted(subjects.keys(), key=lambda x: int(x))
    n           = len(subject_ids)
    print(f"Total subjects      : {n}")

    # ── Subject-level split ───────────────────────────────
    random.shuffle(subject_ids)
    n_train = int(n * args.train_ratio)
    n_val   = int(n * args.val_ratio)
    n_test  = n - n_train - n_val

    train_sids = set(subject_ids[:n_train])
    val_sids   = set(subject_ids[n_train:n_train + n_val])
    test_sids  = set(subject_ids[n_train + n_val:])

    print(f"\nTrain subjects ({len(train_sids)}): {sorted(train_sids, key=int)}")
    print(f"Val   subjects ({len(val_sids)}):  {sorted(val_sids,   key=int)}")
    print(f"Test  subjects ({len(test_sids)}):  {sorted(test_sids,  key=int)}")

    # ── Create output directories ──────────────────────────
    out_root = root / 'split'
    for split in ['train', 'val', 'test']:
        (out_root / split / 'color').mkdir(parents=True, exist_ok=True)
        (out_root / split / 'depth').mkdir(parents=True, exist_ok=True)

    # ── Copy files ────────────────────────────────────────
    counts = {'train': {'real': 0, 'spoof': 0},
              'val':   {'real': 0, 'spoof': 0},
              'test':  {'real': 0, 'spoof': 0}}

    for p in unique:
        sid = parse_subject_id(p.name)
        if   sid in train_sids: split = 'train'
        elif sid in val_sids:   split = 'val'
        else:                   split = 'test'

        label = 'real' if '_real' in p.name else 'spoof'
        copy_sample(p,
                    out_root / split / 'color',
                    out_root / split / 'depth',
                    depth_dirs)
        counts[split][label] += 1

    # ── Summary ───────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Split complete — saved to: {out_root}")
    for split in ['train', 'val', 'test']:
        c = counts[split]
        total = c['real'] + c['spoof']
        print(f"  {split:5s}: real={c['real']:4d}  spoof={c['spoof']:4d}  "
              f"total={total:4d}  "
              f"({total/len(unique)*100:.0f}%)")
    print(f"\nNo subject overlap across splits ✓")
    print(f"\nTraining command:")
    print(f"  python train.py --data_root data/casiafasd/split \\")
    print(f"    --model_size small --epochs 50 --lr 5e-5 \\")
    print(f"    --no_midas --num_workers 0 --batch_size 4 \\")
    print(f"    --img_size 128 --val_split 0.0 \\")
    print(f"    --dropout 0.5 --weight_decay 1e-2")


if __name__ == '__main__':
    main()
