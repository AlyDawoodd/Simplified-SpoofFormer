"""
Quick smoke test — verifies the full dual-stream pipeline WITHOUT a dataset.
Runs: component checks -> forward pass -> backward -> AMP -> metrics -> ONNX.

Run: python test_pipeline.py
Expected: 11/11 checks pass.
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from models.model import (
    spoofformer_small, spoofformer_base,
    ConvStem, MultiScalePatchEmbedding, HybridAttentionBlock,
    ClassificationHead, SpoofFormerStream, DualStreamSpoofFormer,
)
from utils.metrics import evaluate_all

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PASS, FAIL = "✓", "✗"
results = []

def check(name, fn):
    try:
        fn()
        print(f"  {PASS} {name}")
        results.append((name, True))
    except Exception as e:
        print(f"  {FAIL} {name}  ->  {e}")
        results.append((name, False))


print(f"\n{'='*55}")
print(f"  SpoofFormer Dual-Stream Smoke Test  [{DEVICE}]")
print(f"{'='*55}")

# ── [1] Components ────────────────────────────────────

print("\n[1] Model Components")

def test_conv_stem():
    m   = ConvStem(3, 128).to(DEVICE)
    out = m(torch.randn(2, 3, 224, 224).to(DEVICE))
    assert out.shape == (2, 128, 224, 224)

check("ConvStem RGB  (B,3,224,224) -> (B,128,224,224)", test_conv_stem)

def test_conv_stem_depth():
    m   = ConvStem(1, 128).to(DEVICE)
    out = m(torch.randn(2, 1, 224, 224).to(DEVICE))
    assert out.shape == (2, 128, 224, 224)

check("ConvStem Depth (B,1,224,224) -> (B,128,224,224)", test_conv_stem_depth)

def test_patch_embed():
    m            = MultiScalePatchEmbedding(128, 512).to(DEVICE)
    tokens, H, W = m(torch.randn(2, 128, 224, 224).to(DEVICE))
    assert tokens.shape == (2, 784, 512)
    assert H == W == 28

check("MultiScalePatchEmbedding -> (B,784,512)", test_patch_embed)

def test_hybrid_block():
    m   = HybridAttentionBlock(64, 4, window_size=7).to(DEVICE)
    x   = torch.randn(2, 196, 64).to(DEVICE)
    out = m(x)
    assert out.shape == x.shape

check("HybridAttentionBlock preserves shape", test_hybrid_block)

# ── [2] Full Dual-Stream Model ────────────────────────

print("\n[2] Dual-Stream Model")

def test_single_stream():
    s   = SpoofFormerStream(in_channels=3, embed_dim=256, depth=2,
                             num_heads=4).to(DEVICE)
    out = s(torch.randn(2, 3, 224, 224).to(DEVICE))
    assert out.shape == (2, 256), f"Expected (2,256) got {out.shape}"

check("Single stream RGB -> CLS (B, embed_dim)", test_single_stream)

def test_dual_stream_no_midas():
    model = spoofformer_small(use_midas=False).to(DEVICE)
    out   = model(torch.randn(2, 3, 224, 224).to(DEVICE))
    assert out.shape == (2, 2)
    params = model.get_num_params()
    print(f"       Trainable params: {params:,}")

check("Dual-stream forward (no MiDaS, batch=2)", test_dual_stream_no_midas)

def test_dual_stream_precomputed_depth():
    model = spoofformer_small(use_midas=False).to(DEVICE)
    rgb   = torch.randn(2, 3, 224, 224).to(DEVICE)
    depth = torch.randn(2, 1, 224, 224).to(DEVICE)
    out   = model(rgb, depth=depth)
    assert out.shape == (2, 2)

check("Dual-stream with pre-computed depth map", test_dual_stream_precomputed_depth)

# ── [3] Training Step ─────────────────────────────────

print("\n[3] Training Step")

def test_backward():
    model     = spoofformer_small(use_midas=False).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    loss      = criterion(
        model(torch.randn(4, 3, 224, 224).to(DEVICE)),
        torch.randint(0, 2, (4,)).to(DEVICE)
    )
    loss.backward()
    optimizer.step()
    assert not torch.isnan(loss)
    print(f"       Loss: {loss.item():.4f}")

check("Forward + backward + optimizer step", test_backward)

def test_amp():
    if DEVICE.type != 'cuda':
        print("       (skipped — CPU only)")
        return
    from torch.amp import GradScaler, autocast
    model     = spoofformer_small(use_midas=False).to(DEVICE)
    scaler    = GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()
    with autocast('cuda'):
        loss = criterion(
            model(torch.randn(2, 3, 224, 224).to(DEVICE)),
            torch.randint(0, 2, (2,)).to(DEVICE)
        )
    scaler.scale(loss).backward()
    scaler.step(torch.optim.AdamW(model.parameters()))
    scaler.update()

check("AMP (FP16) training step", test_amp)

# ── [4] Metrics ───────────────────────────────────────

print("\n[4] Evaluation Metrics")

def test_metrics():
    np.random.seed(42)
    n      = 500
    y_true = np.random.randint(0, 2, n)
    y_pred = np.clip(y_true * 0.6 + np.random.randn(n) * 0.3, 0, 1)
    m      = evaluate_all(y_true, y_pred, verbose=False)
    assert 0.5 < m['auc'] <= 1.0
    assert 0.0 <= m['eer'] <= 0.5
    print(f"       AUC={m['auc']*100:.1f}%  EER={m['eer']*100:.1f}%  "
          f"ACER={m['acer']*100:.1f}%")

check("APCER / BPCER / ACER / EER / AUC", test_metrics)

# ── [5] ONNX Export ───────────────────────────────────

print("\n[5] ONNX Export")

def test_onnx():
    try:
        import onnx, onnxruntime as ort
    except ImportError:
        print("       (skipped — install onnx onnxruntime)")
        return
    import tempfile, os, torch.onnx

    model = spoofformer_small(use_midas=False).cpu().eval()
    dummy = torch.randn(1, 3, 224, 224)
    depth = torch.randn(1, 1, 224, 224)

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        path = f.name

    torch.onnx.export(
        model, (dummy, depth), path,
        opset_version=17,
        input_names=['rgb', 'depth'],
        output_names=['logits'],
        dynamic_axes={'rgb': {0:'B'}, 'depth': {0:'B'}, 'logits': {0:'B'}},
    )
    size_mb = os.path.getsize(path) / (1024**2)

    # Verify with ONNX Runtime
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    out  = sess.run(None, {'rgb': dummy.numpy(), 'depth': depth.numpy()})[0]
    os.unlink(path)

    assert out.shape == (1, 2)
    print(f"       ONNX file: {size_mb:.1f} MB  |  output: {out.shape}")

check("ONNX export + Runtime inference (dual-input)", test_onnx)

# ── [6] Speed ─────────────────────────────────────────

print("\n[6] Inference Speed")

def test_speed():
    model = spoofformer_base(use_midas=False).to(DEVICE).eval()
    rgb   = torch.randn(1, 3, 224, 224).to(DEVICE)
    depth = torch.randn(1, 1, 224, 224).to(DEVICE)
    for _ in range(3):
        with torch.no_grad(): model(rgb, depth)
    n  = 20
    t0 = time.perf_counter()
    for _ in range(n):
        with torch.no_grad(): model(rgb, depth)
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    ms  = (time.perf_counter() - t0) / n * 1000
    print(f"       {ms:.1f} ms/image  ({1000/ms:.1f} FPS)  [{DEVICE}]")

check("Inference speed (base, 20 runs)", test_speed)

# ── Summary ───────────────────────────────────────────

passed = sum(1 for _, ok in results if ok)
total  = len(results)
print(f"\n{'='*55}")
print(f"  Results: {passed}/{total} checks passed")
if passed == total:
    print("  All checks passed! Ready for training.")
print(f"{'='*55}\n")
if passed < total: sys.exit(1)
