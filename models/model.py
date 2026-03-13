"""
Dual-Stream SpoofFormer: Face Anti-Spoofing with Vision Transformer
Inspired by: "Spoof-formerNet: Face Anti Spoofing with Two-Stage HR-ViT Network"

Architecture:
    RGB Image  ──► ConvStem ──► MultiScalePatchEmbed ──► N × HybridAttnBlock ──┐
                                                                                 ├──► Concat ──► FC ──► real/spoof
    Depth Map  ──► ConvStem ──► MultiScalePatchEmbed ──► N × HybridAttnBlock ──┘

Depth map is generated from RGB using MiDaS monocular depth estimation
(Ranftl et al., 2020) — the standard approach in FAS literature.

Required components (all present in each stream):
    1. Patch Embedding Layer        — MultiScalePatchEmbedding (8x8 + 16x16)
    2. Transformer Encoder Blocks   — HybridAttentionBlock (W-Local + S-Global)
    3. Classification Token [CLS]   — learnable token prepended to sequence
    4. Binary Classification Head   — fused dual-stream CLS -> real/spoof

Simplifications vs. full paper:
    - 2 patch scales instead of 4
    - Standard MSA for S-Global instead of weighted WMSA
    - Sequential encoder blocks instead of 5-level HR multi-branch
    - MiDaS_small for depth (paper does not specify depth model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# DEPTH MAP GENERATOR  (MiDaS — Ranftl et al., 2020)
# ─────────────────────────────────────────────────────────────────────────────

class DepthEstimator(nn.Module):
    """
    Wraps MiDaS monocular depth estimation model.
    Frozen during training — not part of the learnable model.
    Converts (B, 3, H, W) RGB -> (B, 1, H, W) normalized depth map.
    """
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self._model = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        try:
            self._model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small",
                trust_repo=True, verbose=False
            )
            self._model.eval()
            self._model.to(self.device)
            for p in self._model.parameters():
                p.requires_grad = False
            self._loaded = True
            print("[DepthEstimator] MiDaS_small loaded successfully")
        except Exception as e:
            print(f"[DepthEstimator] WARNING: Could not load MiDaS: {e}")
            print("[DepthEstimator] Falling back to Laplacian depth proxy")
            self._loaded = True

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: (B, 3, H, W) ImageNet-normalized tensor
        Returns:
            depth: (B, 1, H, W) in [0, 1]
        """
        self._load()
        B, C, H, W = rgb.shape

        if self._model is not None:
            mean = torch.tensor([0.485, 0.456, 0.406],
                                 device=rgb.device).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225],
                                 device=rgb.device).view(1, 3, 1, 1)
            inp = rgb * std + mean
            inp_resized = F.interpolate(inp, size=(256, 256),
                                        mode='bilinear', align_corners=False)
            depth = self._model(inp_resized)           # (B, 256, 256)
            depth = depth.unsqueeze(1)
            depth = F.interpolate(depth, size=(H, W),
                                   mode='bilinear', align_corners=False)
        else:
            depth = self._laplacian_proxy(rgb)

        return self._normalize(depth)   # (B, 1, H, W)

    def _laplacian_proxy(self, rgb: torch.Tensor) -> torch.Tensor:
        gray   = 0.299*rgb[:,0:1] + 0.587*rgb[:,1:2] + 0.114*rgb[:,2:3]
        kernel = torch.tensor([[0, 1, 0],[1,-4,1],[0, 1, 0]],
                               dtype=torch.float32, device=rgb.device).view(1,1,3,3)
        return F.conv2d(gray, kernel, padding=1).abs()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        B     = x.shape[0]
        x_flat = x.view(B, -1)
        mn    = x_flat.min(dim=1).values.view(B,1,1,1)
        mx    = x_flat.max(dim=1).values.view(B,1,1,1)
        return (x - mn) / (mx - mn + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONVOLUTIONAL STEM  (paper §3.1)
# ─────────────────────────────────────────────────────────────────────────────

class ConvStem(nn.Module):
    """
    Low-level feature extraction before tokenization.
    Paper: conv is more efficient than attention for low-level features.
    Used for both RGB stream (in_channels=3) and Depth stream (in_channels=1).
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MULTI-SCALE PATCH EMBEDDING  (Required component 1)
# ─────────────────────────────────────────────────────────────────────────────

class MultiScalePatchEmbedding(nn.Module):
    """
    Tokenizes at two spatial scales and concatenates embeddings.
    Paper uses 4 scales; we use 2 while preserving the multi-scale concept.

    8x8  patches -> 784 tokens  (fine texture, moire, print artifacts)
    16x16 patches -> 196 tokens  (coarse geometry, depth consistency)
    Both projected to embed_dim/2 and concatenated -> embed_dim per token.
    """
    def __init__(self, in_channels: int = 128, embed_dim: int = 512):
        super().__init__()
        half = embed_dim // 2
        self.small_proj = nn.Conv2d(in_channels, half, kernel_size=8,  stride=8)
        self.large_proj = nn.Conv2d(in_channels, half, kernel_size=16, stride=16)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        small = self.small_proj(x)
        large = self.large_proj(x)
        _, _, Hs, Ws = small.shape

        small = small.flatten(2).transpose(1, 2)   # (B, N_s, half)
        large = large.flatten(2).transpose(1, 2)   # (B, N_l, half)

        large = large.transpose(1, 2)
        large = F.interpolate(large, size=small.shape[1], mode='nearest')
        large = large.transpose(1, 2)

        tokens = torch.cat([small, large], dim=2)  # (B, N_s, embed_dim)
        return self.norm(tokens), Hs, Ws


# ─────────────────────────────────────────────────────────────────────────────
# 3. HYBRID ATTENTION BLOCK  (Required component 2)
# W-Local + S-Global per block  (paper §3.2)
# ─────────────────────────────────────────────────────────────────────────────

class WindowAttention(nn.Module):
    """
    W-Local: window-local self-attention (paper §3.2).
    O(N*w^2) complexity. Captures fine-grained local spoofing artifacts.
    """
    def __init__(self, embed_dim: int, num_heads: int, window_size: int = 7):
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        w   = self.window_size
        pad = (w - N % w) % w
        x = F.pad(x, (0, 0, 0, pad))
        N_pad    = x.shape[1]
        num_wins = N_pad // w
        x_win    = x.view(B, num_wins, w, D).reshape(B * num_wins, w, D)
        out, _   = self.attn(x_win, x_win, x_win)
        out      = out.reshape(B, N_pad, D)
        return out[:, :N, :]


class HybridAttentionBlock(nn.Module):
    """
    Transformer encoder block: W-Local then S-Global attention (paper §3.2).

        x -> LN -> W-Local  -> residual   (captures local texture artifacts)
          -> LN -> S-Global -> residual   (captures global depth inconsistency)
          -> LN -> FFN       -> residual
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 window_size: int = 7, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm_local  = nn.LayerNorm(embed_dim)
        self.local_attn  = WindowAttention(embed_dim, num_heads, window_size)
        self.norm_global = nn.LayerNorm(embed_dim)
        self.global_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=dropout)
        hidden = int(embed_dim * mlp_ratio)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x      = x + self.local_attn(self.norm_local(x))
        g, _   = self.global_attn(self.norm_global(x),
                                   self.norm_global(x),
                                   self.norm_global(x))
        x      = x + g
        x      = x + self.ffn(self.norm_ffn(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE STREAM  (used for both RGB and Depth)
# Contains required components 1, 2, 3
# ─────────────────────────────────────────────────────────────────────────────

class SpoofFormerStream(nn.Module):
    """
    One complete ViT stream with all required components:
        1. Patch Embedding  — MultiScalePatchEmbedding
        2. Encoder Blocks   — N x HybridAttentionBlock
        3. CLS Token        — learnable classification token + positional embed
    Returns the CLS token representation for fusion.
    """
    def __init__(self, in_channels: int, img_size: int = 224,
                 stem_channels: int = 128, embed_dim: int = 512,
                 depth: int = 8, num_heads: int = 8, window_size: int = 7,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()

        # Component 1: Patch Embedding
        self.conv_stem   = ConvStem(in_channels, stem_channels)
        self.patch_embed = MultiScalePatchEmbedding(stem_channels, embed_dim)
        num_patches      = (img_size // 8) ** 2   # 784 for 224x224

        # Component 3: CLS Token + Positional Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(dropout)

        # Component 2: Transformer Encoder Blocks
        self.encoder = nn.ModuleList([
            HybridAttentionBlock(embed_dim, num_heads, window_size,
                                 mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B   = x.shape[0]
        x   = self.conv_stem(x)                    # (B, stem_ch, H, W)
        x, H, W = self.patch_embed(x)              # (B, N, embed_dim)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)           # (B, N+1, embed_dim)
        x   = self.pos_drop(x + self.pos_embed)
        for block in self.encoder:
            x = block(x)
        x   = self.norm(x)
        return x[:, 0]                             # (B, embed_dim) — CLS token


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLASSIFICATION HEAD  (Required component 4)
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    Binary classification head (real vs spoof).
    Fuses CLS tokens from RGB and Depth streams via concatenation.
    in_dim = embed_dim * 2.
    """
    def __init__(self, in_dim: int, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.head = nn.Sequential(
            nn.Linear(in_dim, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),   nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(x))   # (B, 2)


# ─────────────────────────────────────────────────────────────────────────────
# DUAL-STREAM SPOOFFORMER  — Main Model
# ─────────────────────────────────────────────────────────────────────────────

class DualStreamSpoofFormer(nn.Module):
    """
    Dual-stream SpoofFormer-style face anti-spoofing model.

    Mirrors paper architecture:
        RGB   stream -> CLS_rgb
        Depth stream -> CLS_depth
        concat(CLS_rgb, CLS_depth) -> ClassificationHead -> real/spoof

    Depth map generated automatically from RGB via MiDaS (use_midas=True)
    or as grayscale fallback (use_midas=False, faster, slightly lower accuracy).
    """
    def __init__(self, img_size: int = 224, stem_channels: int = 128,
                 embed_dim: int = 512, depth: int = 8, num_heads: int = 8,
                 window_size: int = 7, mlp_ratio: float = 4.0,
                 dropout: float = 0.1, num_classes: int = 2,
                 use_midas: bool = True):
        super().__init__()
        self.use_midas = use_midas
        self.depth_gen = None   # lazy init on first forward

        stream_kwargs = dict(
            img_size=img_size, stem_channels=stem_channels,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            window_size=window_size, mlp_ratio=mlp_ratio, dropout=dropout,
        )

        self.rgb_stream   = SpoofFormerStream(in_channels=3, **stream_kwargs)
        self.depth_stream = SpoofFormerStream(in_channels=1, **stream_kwargs)
        self.head         = ClassificationHead(embed_dim * 2, num_classes, dropout)

    def _get_depth(self, rgb: torch.Tensor) -> torch.Tensor:
        if not self.use_midas:
            # Simple grayscale — no external dependency
            return (0.299*rgb[:,0:1] + 0.587*rgb[:,1:2] + 0.114*rgb[:,2:3])
        if self.depth_gen is None:
            self.depth_gen = DepthEstimator(device=str(rgb.device))
        return self.depth_gen(rgb)

    def forward(self, rgb: torch.Tensor,
                depth: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            rgb:   (B, 3, H, W) — ImageNet-normalized RGB face image
            depth: (B, 1, H, W) — optional pre-computed depth map
        Returns:
            logits: (B, 2)
        """
        if depth is None:
            depth = self._get_depth(rgb)

        cls_rgb   = self.rgb_stream(rgb)               # (B, embed_dim)
        cls_depth = self.depth_stream(depth)           # (B, embed_dim)
        fused     = torch.cat([cls_rgb, cls_depth], dim=1)  # (B, embed_dim*2)
        return self.head(fused)                        # (B, 2)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def spoofformer_small(**kwargs) -> DualStreamSpoofFormer:
    """~50M trainable params"""
    return DualStreamSpoofFormer(embed_dim=256, depth=6, num_heads=4, **kwargs)

def spoofformer_base(**kwargs) -> DualStreamSpoofFormer:
    """~100M trainable params — recommended"""
    return DualStreamSpoofFormer(embed_dim=512, depth=8, num_heads=8, **kwargs)

def spoofformer_large(**kwargs) -> DualStreamSpoofFormer:
    """~200M trainable params"""
    return DualStreamSpoofFormer(embed_dim=768, depth=12, num_heads=12, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = spoofformer_base(use_midas=False).to(device)
    dummy  = torch.randn(2, 3, 224, 224).to(device)

    with torch.no_grad():
        out = model(dummy)

    print(f"Output shape      : {out.shape}")
    print(f"Trainable params  : {model.get_num_params():,}")
    print(f"Device            : {device}")
    print("Smoke test PASSED ✓")
