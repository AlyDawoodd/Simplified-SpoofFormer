"""
Inference script for Dual-Stream SpoofFormer.

Features:
  - Single image inference (RGB + auto depth)
  - ONNX export (dual-input: rgb + depth)
  - Speed benchmarking: PyTorch vs ONNX Runtime

Usage:
    # Benchmark + ONNX export
    python inference.py --checkpoint checkpoints/best_model.pth --export_onnx --benchmark

    # Single image inference
    python inference.py --image path/to/face.jpg --checkpoint checkpoints/best_model.pth

    # ONNX inference
    python inference.py --image path/to/face.jpg --onnx_model spoofformer.onnx --depth path/to/depth.jpg
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from models.model import spoofformer_small, spoofformer_base, spoofformer_large


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_rgb(img_path: str, img_size: int = 128) -> torch.Tensor:
    """Load and preprocess RGB face image."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    pil = TF.resize(pil, (img_size, img_size))
    t   = TF.to_tensor(pil)
    t   = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return t.unsqueeze(0)   # (1, 3, H, W)


def preprocess_depth(depth_path: str, img_size: int = 128) -> torch.Tensor:
    """Load and preprocess depth map."""
    img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read depth: {depth_path}")
    pil = Image.fromarray(img).convert('L')
    pil = TF.resize(pil, (img_size, img_size))
    t   = TF.to_tensor(pil)   # (1, H, W)
    return t.unsqueeze(0)     # (1, 1, H, W)


def rgb_to_depth_grayscale(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """Simple grayscale depth proxy when no depth map is available."""
    # Unnormalize RGB first
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    rgb  = rgb_tensor * std + mean
    gray = 0.299*rgb[:,0:1] + 0.587*rgb[:,1:2] + 0.114*rgb[:,2:3]
    return gray   # (1, 1, H, W)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

def load_model(checkpoint_path: str, model_size: str = 'small',
               img_size: int = 128, use_midas: bool = False,
               device: str = 'cpu'):
    builders = {'small': spoofformer_small,
                'base':  spoofformer_base,
                'large': spoofformer_large}
    model = builders[model_size](img_size=img_size, use_midas=use_midas)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval().to(device)
    print(f"[Model] Loaded {model_size} from epoch {ckpt.get('epoch','?')}")
    print(f"[Model] Val ACER at save: {ckpt.get('best_acer',0)*100:.2f}%")
    return model


# ─────────────────────────────────────────────
# PYTORCH INFERENCE
# ─────────────────────────────────────────────

def pytorch_predict(model, rgb: torch.Tensor,
                    depth: torch.Tensor, device: str) -> dict:
    rgb   = rgb.to(device)
    depth = depth.to(device)
    with torch.no_grad():
        logits = model(rgb, depth)
    probs      = torch.softmax(logits, dim=1)
    spoof_prob = probs[0, 1].item()
    real_prob  = probs[0, 0].item()
    return {
        'label':      'SPOOF' if spoof_prob > 0.5 else 'REAL',
        'real_prob':  real_prob,
        'spoof_prob': spoof_prob,
        'confidence': max(real_prob, spoof_prob),
    }


# ─────────────────────────────────────────────
# ONNX EXPORT
# ─────────────────────────────────────────────

def export_onnx(model, img_size: int = 128,
                output_path: str = 'spoofformer.onnx'):
    import torch.onnx
    model.cpu().eval()
    dummy_rgb   = torch.randn(1, 3, img_size, img_size)
    dummy_depth = torch.randn(1, 1, img_size, img_size)

    torch.onnx.export(
        model,
        (dummy_rgb, dummy_depth),
        output_path,
        opset_version=17,
        input_names=['rgb', 'depth'],
        output_names=['logits'],
        dynamic_axes={
            'rgb':    {0: 'batch'},
            'depth':  {0: 'batch'},
            'logits': {0: 'batch'},
        },
    )
    size_mb = Path(output_path).stat().st_size / (1024**2)
    print(f"[ONNX] Exported to: {output_path}  ({size_mb:.1f} MB)")

    # Verify with ONNX
    try:
        import onnx
        onnx.checker.check_model(output_path)
        print("[ONNX] Model verified ✓")
    except ImportError:
        pass

    return output_path



# ─────────────────────────────────────────────
# TORCHSCRIPT EXPORT
# ─────────────────────────────────────────────

def export_torchscript(model, img_size: int = 128,
                       output_path: str = 'spoofformer_scripted.pt'):
    """
    Export model to TorchScript via torch.jit.trace.
    Enables deployment in C++ environments and PyTorch Mobile
    without Python dependency.
    """
    model.cpu().eval()
    dummy_rgb   = torch.randn(1, 3, img_size, img_size)
    dummy_depth = torch.randn(1, 1, img_size, img_size)

    with torch.no_grad():
        traced = torch.jit.trace(model, (dummy_rgb, dummy_depth))

    traced.save(output_path)
    size_mb = Path(output_path).stat().st_size / (1024**2)
    print(f"[TorchScript] Exported to: {output_path}  ({size_mb:.1f} MB)")
    return output_path, traced





# ─────────────────────────────────────────────
# BENCHMARK
# ─────────────────────────────────────────────

def benchmark(model=None, onnx_path=None, scripted_model=None,
              img_size: int = 128, device: str = 'cpu',
              n_runs: int = 50):
    dummy_rgb   = torch.randn(1, 3, img_size, img_size)
    dummy_depth = torch.randn(1, 1, img_size, img_size)
    results     = {}

    # ── PyTorch ───────────────────────────────
    if model is not None:
        model.eval()
        rgb_d   = dummy_rgb.to(device)
        depth_d = dummy_depth.to(device)

        for _ in range(5):   # warmup
            with torch.no_grad(): model(rgb_d, depth_d)
        if device == 'cuda': torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            with torch.no_grad(): model(rgb_d, depth_d)
            if device == 'cuda': torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        pt_ms = float(np.mean(times))
        results['pytorch_ms']  = pt_ms
        results['pytorch_fps'] = 1000 / pt_ms
        print(f"[Benchmark] PyTorch  ({device}):   "
              f"{pt_ms:.1f} ms/image  ({1000/pt_ms:.1f} FPS)")

    # ── ONNX Runtime (CPU + GPU if available) ─
    if onnx_path and Path(onnx_path).exists():
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()

            # Try CUDA first, fall back to CPU
            if 'CUDAExecutionProvider' in available:
                providers  = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                ort_device = 'GPU'
            else:
                providers  = ['CPUExecutionProvider']
                ort_device = 'CPU'

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level =                 ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4

            sess    = ort.InferenceSession(onnx_path,
                                           sess_options=sess_options,
                                           providers=providers)
            rgb_np   = dummy_rgb.numpy().astype(np.float32)
            depth_np = dummy_depth.numpy().astype(np.float32)

            for _ in range(10):   # warmup
                sess.run(None, {'rgb': rgb_np, 'depth': depth_np})

            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                sess.run(None, {'rgb': rgb_np, 'depth': depth_np})
                times.append((time.perf_counter() - t0) * 1000)

            ort_ms = float(np.mean(times))
            results['onnx_ms']  = ort_ms
            results['onnx_fps'] = 1000 / ort_ms
            print(f"[Benchmark] ONNX Runtime ({ort_device}): "
                  f"{ort_ms:.1f} ms/image  ({1000/ort_ms:.1f} FPS)")

            if 'pytorch_ms' in results:
                speedup = results['pytorch_ms'] / ort_ms
                results['speedup'] = speedup
                label = "faster" if speedup > 1 else "slower"
                print(f"[Benchmark] ONNX vs PyTorch: {speedup:.2f}x {label}")
        except ImportError:
            print("[Benchmark] onnxruntime not installed — skipping ONNX benchmark")


    # ── TorchScript ───────────────────────────
    if scripted_model is not None:
        scripted_model.eval()
        rgb_d   = dummy_rgb.to(device)
        depth_d = dummy_depth.to(device)

        for _ in range(5):
            with torch.no_grad(): scripted_model(rgb_d, depth_d)
        if device == 'cuda': torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            with torch.no_grad(): scripted_model(rgb_d, depth_d)
            if device == 'cuda': torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        ts_ms = float(np.mean(times))
        results['torchscript_ms']  = ts_ms
        results['torchscript_fps'] = 1000 / ts_ms
        print(f"[Benchmark] TorchScript ({device}): "
              f"{ts_ms:.1f} ms/image  ({1000/ts_ms:.1f} FPS)")



    return results


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Dual-Stream SpoofFormer Inference")
    p.add_argument('--image',        default=None,  help='RGB face image path')
    p.add_argument('--depth',        default=None,  help='Depth map path (optional)')
    p.add_argument('--checkpoint',   default=None,  help='.pth checkpoint path')
    p.add_argument('--model_size',   default='small',
                   choices=['small', 'base', 'large'])
    p.add_argument('--img_size',     default=128,   type=int)
    p.add_argument('--no_midas',     action='store_true',
                   help='Use grayscale depth proxy instead of MiDaS')
    p.add_argument('--export_onnx',       action='store_true')
    p.add_argument('--export_torchscript', action='store_true')

    p.add_argument('--torchscript_output', default='spoofformer_scripted.pt')

    p.add_argument('--onnx_output',  default='spoofformer.onnx')
    p.add_argument('--onnx_model',   default=None,  help='Run inference with ONNX model')
    p.add_argument('--benchmark',    action='store_true')
    p.add_argument('--n_runs',       default=50,    type=int)
    p.add_argument('--device',       default='cuda' if torch.cuda.is_available()
                                             else 'cpu')
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*50}")
    print(f"  Dual-Stream SpoofFormer Inference")
    print(f"  device={args.device} | img_size={args.img_size}")
    print(f"{'='*50}\n")

    # ── Load model ────────────────────────────
    model = None
    if args.checkpoint:
        model = load_model(
            args.checkpoint,
            model_size=args.model_size,
            img_size=args.img_size,
            use_midas=not args.no_midas,
            device=args.device,
        )

    # ── Single image inference ────────────────
    if args.image:
        rgb = preprocess_rgb(args.image, args.img_size)

        if args.depth:
            depth = preprocess_depth(args.depth, args.img_size)
            print(f"[Depth] Loaded from: {args.depth}")
        elif not args.no_midas:
            # Generate depth with MiDaS at runtime
            print("[Depth] Generating depth map with MiDaS...")
            import torch.nn.functional as F_
            midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small",
                                   trust_repo=True, verbose=False)
            midas.eval().to(args.device)
            for p in midas.parameters():
                p.requires_grad = False

            img_bgr  = cv2.imread(args.image)
            img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
            t        = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0).to(args.device)
            t_res    = F_.interpolate(t, size=(256,256), mode="bilinear", align_corners=False)
            with torch.no_grad():
                d = midas(t_res).squeeze().cpu().numpy()
            h, w = img_bgr.shape[:2]
            d    = cv2.resize(d, (w, h))
            d    = d - d.min()
            if d.max() > 0: d = d / d.max()
            d    = (d * 255).astype("uint8")
            from PIL import Image as PILImage
            depth_pil = PILImage.fromarray(d).convert("L")
            import torchvision.transforms.functional as TF_
            depth_pil = TF_.resize(depth_pil, (args.img_size, args.img_size))
            depth     = TF_.to_tensor(depth_pil).unsqueeze(0)
            print("[Depth] MiDaS depth map generated ✓")
        else:
            depth = rgb_to_depth_grayscale(rgb)
            print("[Depth] Using grayscale proxy (--no_midas flag set)")

        if model:
            result = pytorch_predict(model, rgb, depth, args.device)
            print(f"\n[Result] Label     : {result['label']}")
            print(f"[Result] Real prob  : {result['real_prob']*100:.1f}%")
            print(f"[Result] Spoof prob : {result['spoof_prob']*100:.1f}%")
            print(f"[Result] Confidence : {result['confidence']*100:.1f}%")

    # ── ONNX export ───────────────────────────
    onnx_path = None
    if args.export_onnx and model:
        onnx_path = export_onnx(model, args.img_size, args.onnx_output)
        model.to(args.device)  # move back to device after ONNX export

    # ── TorchScript export ────────────────────
    scripted_model = None
    if args.export_torchscript and model:
        ts_path, scripted_model = export_torchscript(
            model, args.img_size, args.torchscript_output)
        model.to(args.device)          # restore original model to device
        scripted_model.to(args.device) # move scripted model to device too

    if args.onnx_model:
        onnx_path = args.onnx_model

    # ── Benchmark ─────────────────────────────
    if args.benchmark:
        print(f"\n[Benchmark] {args.n_runs} runs, img_size={args.img_size}")
        results = benchmark(
            model=model,
            onnx_path=onnx_path,
            scripted_model=scripted_model,
            img_size=args.img_size,
            device=args.device,
            n_runs=args.n_runs,
        )
        # Print summary table
        sep = "="*50
        print(f"\n[Summary] Optimization Results")
        print(sep)
        labels = {
            'pytorch_fps':     'PyTorch (GPU)',
            'torchscript_fps': 'TorchScript (GPU)',
            'onnx_fps':        'ONNX Runtime',
        }
        for key, label in labels.items():
            if key in results:
                print(f"  {label:<25} {results[key]:.1f} FPS")
        print(sep)
        print("  TorchScript recommended for GPU/C++ deployment")
        print("  ONNX recommended for cross-platform deployment")


if __name__ == '__main__':
    main()
