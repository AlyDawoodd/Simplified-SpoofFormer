# Dual-Stream SpoofFormer — Face Anti-Spoofing

A simplified implementation of the SpoofFormer architecture for face anti-spoofing. This document covers the full development journey including problems encountered and how they were resolved.

---

## Final Results

| Metric | Value |
|--------|-------|
| AUC    | 99.39% |
| EER    | 2.44% |
| APCER  | 0.47% |
| BPCER  | 4.41% |
| ACER   | 2.44% |



---

## Architecture

The model follows the dual-stream design from *"Spoof-formerNet: Face Anti Spoofing with Two-Stage HR-ViT Network"*:
```
                   RGB IMAGE                    DEPTH MAP (MiDaS)
                       │                              │
                    ConvStem                       ConvStem
                       │                              │
          MultiScale Patch Embedding     MultiScale Patch Embedding
                       │                              │
           8× HybridAttentionBlocks        8× HybridAttentionBlocks
                       │                              │
                    CLS Token                       CLS Token
                         \                          /
                          \                        /
                           \                      /
                            └── Concatenate ────┘
                                    │
                           Fully Connected Layer
                                    │
                               Real / Spoof
```
**Key components:**
- **ConvStem** — 2-layer Conv-BN-GELU before transformer (paper §3.1)
- **MultiScalePatchEmbedding** — 8×8 + 16×16 patches concatenated
- **HybridAttentionBlock** — WindowAttention (local) + MultiheadAttention (global) per block
- **CLS token** + positional embedding per stream
- **ClassificationHead** — fused dual-stream CLS → real/spoof

**Justified simplifications vs paper:**
- 2 patch scales instead of 4 (core concept preserved)
- Sequential encoder blocks instead of 5-level HR multi-branch
- Standard MSA for global attention instead of weighted WMSA

**Model sizes:**
- `small` — ~24M params (used for this task)
- `base` — ~100M params
- `large` — ~200M params

---

## Installation

```bash
git clone <repo>
cd spoofformer
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

---



## Dataset

I used `minhnh2107/casiafasd` from Kaggle (74MB).

### Download

```bash
pip install kaggle
kaggle datasets download minhnh2107/casiafasd
unzip casiafasd.zip -d data/casiafasd
```

### Dataset structure after download
```
data/casiafasd/
    train_img/train_img/color/   ← RGB frames
    train_img/train_img/depth/   ← pre-computed depth maps
    test_img/test_img/color/
    test_img/test_img/depth/
```

**Label encoding:** filenames contain `_real` or `_fake` suffix.

### Dataset limitations

CASIA-FASD used covers only **print attacks** under controlled conditions — one camera, controlled lighting, frontal poses. It does not cover:

- **3D mask attacks** — silicone or paper masks worn on the face
- **Diverse demographics** — limited subject variety compared to datasets like CelebA
- **Varied lighting and angles** — real-world eKYC captures happen in uncontrolled environments
- **Digital injection attacks** — virtual cameras or deepfakes

For production eKYC deployment, training on a more diverse dataset such as OULU-NPU or MSU-MFSD would significantly improve generalization. For 3D mask attack detection specifically, **DPT (Dense Prediction Transformer)** for depth estimation would be more suitable than MiDaS, as it captures finer surface geometry needed to distinguish a silicone mask from a real face.

---

## Problem 1: Data Leakage in Original Dataset Split

**What happened:** The Kaggle upload of CASIA-FASD had a corrupted train/test split — 14,033 images appeared in both train and test folders out of 2,660 unique images. Training immediately produced AUC=100%, ACER=0% which was suspicious.

**Root cause:** The dataset uploader likely copied files from both splits into both folders by mistake, causing near-complete overlap.

**How I fixed it:** I wrote `prepare_dataset.py` which performs a clean **subject-level 70/20/10 split** with zero overlap:

```bash
python prepare_dataset.py --data_root data/casiafasd
```

- Groups images by subject ID (parsed from filename prefix e.g. `8_1.avi_50_real.jpg` → subject `8`)
- 30 total subjects: 21 train / 6 val / 3 test
- No subject appears in more than one split
- Result: train=1860 images, val=503, test=~400



---

## Problem 2: Trivially Black Depth Maps for Spoof Images

**What happened:** The original CASIA-FASD depth maps were captured with an **IR/structured light sensor** (likely Intel RealSense). Real faces produce rich 3D geometry. But printed photos held in front of the sensor return completely black depth maps — the sensor physically cannot detect depth on flat paper.

This made the classification problem trivial — the depth stream could achieve 100% accuracy just by checking if the depth map was black or not. The model learned nothing meaningful.

**How I fixed it:** I generated new depth maps for ALL images (real and spoof) using **MiDaS** — a neural network depth estimator that works on regular RGB images:

```bash
python generate_depth.py --data_root data/casiafasd/split
```

This replaced the hardware-captured depth maps with MiDaS-estimated depth maps. Now:
- Real faces → MiDaS estimates natural 3D depth variation
- Spoof faces → MiDaS estimates some texture depth from the photo surface

The problem became genuinely challenging. Initial AUC dropped from 100% to ~89%, confirming the model now had to actually learn discriminative features.

Original depth maps are backed up to `depth_original/` in each split folder.

**Why MiDaS over DPT:** I chose MiDaS_small over DPT (Dense Prediction Transformer) for depth estimation. DPT produces higher quality depth maps but requires ~400MB and ~35ms per image. MiDaS_small requires only ~30MB and ~5ms per image. Given that CASIA-FASD contains only print (no 3D mask attacks), MiDaS provides sufficient depth discrimination at a fraction of the cost. For a production eKYC system running on regular phone cameras without IR sensors, MiDaS is the practical choice. For deployments requiring 3D mask detection, DPT would be the better option.

---

## Problem 3: Overfitting on Small Dataset

With only 1,860 training images, the model overfitted quickly. Three approaches were tried:

### Attempt 1: Full augmentation + oversampling (failed)
- Augmentation: random crop, flip, rotation ±15°, blur, erasing, Gaussian noise
- Oversampling real class to match spoof count
- Result: val AUC peaked at 93% then degraded — augmentation was too aggressive, corrupting the MiDaS depth signal

### Attempt 2: Reduced augmentation + no oversampling (failed)
- Lighter augmentation: flip + 10° rotation only
- Still overfitted at epoch 3 (val loss jumped from 0.47 → 0.62)

### Attempt 3: No augmentation + strong regularization (success)
- No augmentation — pure resize + normalize
- `dropout=0.4`, `weight_decay=1e-2`
- Class-weighted CrossEntropyLoss to handle real/spoof imbalance without oversampling
- Result: stable training, AUC improved consistently to 97-99%

**Key insight:** For datasets under ~2000 images, augmentation can hurt more than help — especially when one stream (depth) carries structured signal that augmentation corrupts. Strong regularization is more reliable at this scale.

---

## Problem 4: Learning Rate Tuning

Several learning rates were tried before finding the optimum:

| LR | Warmup | Result |
|----|--------|--------|
| `5e-5` | 5 epochs | Overfitting from epoch 2 — warmup caused aggressive early updates |
| `2e-5` | 0 epochs | Plateau at AUC~91%, no improvement after epoch 4 |
| `1e-5` | 0 epochs | Slow convergence, best ACER=5.84% at epoch 14 |
| `3e-5` | 2 epochs | **Best** — converged at epoch 5, ACER=4.64%, early stop at epoch 10 |

`lr=3e-5` with 2-epoch warmup + cosine decay was the sweet spot — fast enough to converge in ~5 epochs, stable enough not to overfit.

---

## Training

### Step 1 — Prepare dataset (run once)
```bash
python prepare_dataset.py --data_root data/casiafasd
```

### Step 2 — Generate MiDaS depth maps (run once)
```bash
python generate_depth.py --data_root data/casiafasd/split
```

### Step 3 — Train
```bash
python train.py \
  --data_root data/casiafasd/split \
  --model_size small \
  --epochs 20 \
  --lr 3e-5 \
  --no_midas \
  --num_workers 0 \
  --batch_size 8 \
  --img_size 128 \
  --val_split 0.0 \
  --dropout 0.4 \
  --weight_decay 1e-2 \
  --warmup_epochs 2 \
  --aug_level none \
  --no_oversample \
  --early_stop 5
```

**Key training decisions:**
- `--img_size 128` — reduces token count from 784 to 256, 9x less attention compute vs 224×224
- `--model_size small` — 24M params, right-sized for 1860 training samples
- `--early_stop 5` — automatically stops when no improvement for 5 epochs, saved ~50% compute
- `--no_midas` — load pre-generated MiDaS depth from disk (faster than runtime inference)
- `--num_workers 0` — required on Windows to avoid multiprocessing errors

Training stopped automatically at epoch 10. Best checkpoint from epoch 5 saved to `checkpoints/best_model.pth`.

---

## Evaluation

```bash
python evaluate.py \
  --data_root data/casiafasd/split \
  --checkpoint checkpoints/best_model.pth \
  --no_midas \
  --split test
```

Outputs:
- Full metrics (AUC, EER, APCER, BPCER, ACER) printed to console
- ROC curve → `evaluation_results/roc_curve.png`
- Full report → `evaluation_results/metrics.json`

---

## Inference & Optimization

```bash
# Export ONNX + TorchScript + benchmark
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --no_midas \
  --export_onnx \
  --export_torchscript \
  --benchmark

# Single image inference (MiDaS generates depth automatically)

# Linux/Mac:
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --image path/to/face.jpg

# Windows:
python inference.py --checkpoint checkpoints/best_model.pth --image path/to/face.jpg

# Note: first run downloads MiDaS_small (~30MB) automatically via torch.hub
```

> **Note:** `--no_midas` is only for dataset evaluation where depth maps already exist on disk.
> For inference on any arbitrary image, omit `--no_midas` — MiDaS generates depth automatically.

### Benchmark Results

| Format | Device | Latency | FPS |
|--------|--------|---------|-----|
| PyTorch | GPU (CUDA) | 23.0ms | 43.5 |
| TorchScript | GPU (CUDA) | 19.4ms | 51.7 |
| ONNX FP32 | CPU | 107ms | 9.3 |

### Optimization techniques implemented

**1. TorchScript Export (91.2MB)**
Compiles the model to a static C++ computation graph via `torch.jit.trace`, eliminating
Python interpreter overhead at inference time. Achieved the lowest latency at 19.4ms/51.7 FPS
on GPU — 19% faster than standard PyTorch. Suitable for PyTorch Mobile, C++ server deployment,
and any latency-critical production environment.

**2. ONNX Export (91.5MB)**
Exports the dual-stream model with both RGB and depth as named inputs for clean API integration.
Removes PyTorch dependency entirely — enables deployment eKYC SDK on Android, iOS,
and server environments via ONNX Runtime. Graph optimization level `ORT_ENABLE_ALL` is applied
at runtime to fuse layers and eliminate redundant operations automatically. CUDA provider is
used when available, falling back to CPU otherwise.

**3. Input Resolution Optimization (128×128)**
Reducing input resolution from the standard 224×224 to 128×128 cuts the token sequence from
784 to 256 tokens, decreasing global self-attention complexity by 9x (O(N²) scaling). This
was the single largest speed improvement — enabling 50+ FPS while maintaining AUC=99.39%.

**4. FP16 Mixed Precision Training (AMP)**
Automatic Mixed Precision throughout training reduced GPU memory by ~50%, enabling batch_size=8
on 6GB VRAM. Label smoothing (0.1) and gradient clipping (max_norm=1.0) applied alongside AMP
for training stability.

**5. Early Stopping**
Training automatically stopped at epoch 10 after no improvement for 5 consecutive epochs,
saving ~50% of planned compute while preserving the best checkpoint from epoch 5.

---

## Production Considerations

### Threshold tuning for finance/eKYC

The default decision threshold is 0.5. In financial applications, letting a spoof through (APCER) is far more dangerous than occasionally blocking a real user (BPCER). The threshold should be lowered to prioritize spoof detection:

| Threshold | Effect |
|-----------|--------|
| 0.5 (default) | Balanced — APCER=0.47%, BPCER=4.41% |
| 0.3 | Catches more spoofs — lower APCER, higher BPCER |
| 0.2 | Maximum spoof sensitivity — very few attacks get through |

In this implementation, real faces were classified with ~98% confidence and spoof faces with ~84% confidence, indicating strong class separation. Lowering the threshold to 0.3 would catch essentially all spoofs while adding only minimal false alarms on real users.

### Scaling to production

For a production eKYC system beyond CASIA-FASD:
- **Larger dataset** — OULU-NPU or MSU-MFSD for diverse demographics and lighting conditions
- **3D mask attacks** — replace MiDaS with DPT for finer depth geometry needed to detect silicone masks


---

## Quick Test (Notebook)
## Download Checkpoint
Download `best_model.pth` from [Releases](https://github.com/AlyDawoodd/Simplified-SpoofFormer/releases) 
and place it in `checkpoints/`.
```bash
jupyter notebook test.ipynb
```

Place 4 images in `test_samples/` folder:
```
test_samples/
    1real.jpg    ← real face from test split
    1fake.jpg    ← spoof face from test split
    2real.jpg    ← real face from test split
    2fake.jpg    ← spoof face from test split
```

Displays RGB, MiDaS depth map, confidence bar chart, and correct/wrong verdict for each sample.

---

## Project Structure

```
spoofformer/
├── models/
│   └── model.py              ← DualStreamSpoofFormer — ConvStem, MultiScalePatchEmbed,
│                                HybridAttentionBlock, ClassificationHead
├── data/                     (created after dataset download)
│   └── dataset.py            ← CASIAFASDDataset, SyncedTransform, build_dataloaders
├── utils/
│   └── metrics.py            ← compute_auc, compute_eer, compute_apcer/bpcer/acer
├── configs/
│   └── default.yaml          ← Final optimal configuration with comments
├── checkpoints/              (created after training)
│   └── best_model.pth        ← Best checkpoint (epoch 5, Val ACER=4.64%)
├── evaluation_results/
│   ├── roc_curve.png         ← ROC curve (AUC=99.39%)
│   └── metrics.json          ← Full metrics report
├── test_samples/             ← Place test images here for test.ipynb
├── train.py                  ← Training loop — AMP, early stopping, cosine LR, class weights
├── evaluate.py               ← Full evaluation on test split with ROC curve
├── inference.py              ← TorchScript + ONNX export, benchmark, single image inference
├── prepare_dataset.py        ← Subject-level 70/20/10 split — zero data leakage
├── generate_depth.py         ← MiDaS_small depth map generation for all splits
├── test.ipynb                ← Visual inference test — 2 real + 2 spoof with depth maps
├── test_pipeline.py          ← Smoke tests — 12/12 passing
├── paper_summary.docx        ← 1-page paper summary
├── requirements.txt
└── README.md
```

---

## Requirements

```
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
opencv-python>=4.8.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
onnx>=1.15.0
onnxruntime>=1.16.0
tqdm>=4.65.0
timm>=0.9.0
Pillow>=10.0.0
pyyaml>=6.0
jupyter>=1.0.0
```

---

## Comparison to Paper

| Metric | Full Paper (HR-ViT) | This Implementation |
|--------|--------------------|-----------------------|
| AUC    | 99.22%             | **99.39%**            |
| ACER   | 0.81%              | 2.44%                 |
| EER    | ~1%                | 2.44%                 |
