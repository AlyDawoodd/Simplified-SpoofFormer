"""
Anti-spoofing evaluation metrics.

ISO/IEC 30107-3 standard metrics:
  - APCER  : Attack Presentation Classification Error Rate
  - BPCER  : Bona Fide Presentation Classification Error Rate
  - ACER   : Average Classification Error Rate = (APCER + BPCER) / 2
  - EER    : Equal Error Rate
  - AUC    : Area Under the ROC Curve
"""

from typing import Tuple as Tuple_

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_apcer(y_true: np.ndarray, y_pred: np.ndarray,
                  threshold: float = 0.5) -> float:
    """
    APCER: fraction of spoof samples classified as real.
    Lower is better. Measures security risk.

    Args:
        y_true: ground truth labels  (0=real, 1=spoof)
        y_pred: predicted spoof probability scores
        threshold: decision threshold

    Returns:
        APCER in [0, 1]
    """
    spoof_mask = y_true == 1
    if spoof_mask.sum() == 0:
        return 0.0
    spoof_pred = y_pred[spoof_mask]
    # Spoof classified as real = predicted below threshold
    misclassified = (spoof_pred < threshold).sum()
    return float(misclassified) / float(spoof_mask.sum())


def compute_bpcer(y_true: np.ndarray, y_pred: np.ndarray,
                  threshold: float = 0.5) -> float:
    """
    BPCER: fraction of genuine samples classified as spoof.
    Lower is better. Measures user experience impact.

    Args:
        y_true: ground truth labels  (0=real, 1=spoof)
        y_pred: predicted spoof probability scores
        threshold: decision threshold

    Returns:
        BPCER in [0, 1]
    """
    real_mask = y_true == 0
    if real_mask.sum() == 0:
        return 0.0
    real_pred = y_pred[real_mask]
    # Real classified as spoof = predicted above threshold
    misclassified = (real_pred >= threshold).sum()
    return float(misclassified) / float(real_mask.sum())


def compute_acer(apcer: float, bpcer: float) -> float:
    """
    ACER: balanced summary of APCER and BPCER.
    ACER = (APCER + BPCER) / 2
    """
    return (apcer + bpcer) / 2.0


def compute_eer(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple_[float, float]:
    """
    EER: threshold-independent operating point where APCER == BPCER.
    Lower is better. Used for cross-protocol comparison.

    Returns:
        (eer_value, eer_threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    fnr = 1.0 - tpr

    # Find point where FPR ≈ FNR
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])
    return eer, eer_threshold


def compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    AUC: overall discrimination ability across all thresholds.
    Higher is better. 1.0 = perfect, 0.5 = random.
    """
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_pred))


def find_best_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Find threshold that minimizes ACER on the current set.
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    best_acer = float('inf')
    best_thresh = 0.5

    for t in thresholds:
        apcer = compute_apcer(y_true, y_pred, t)
        bpcer = compute_bpcer(y_true, y_pred, t)
        acer  = compute_acer(apcer, bpcer)
        if acer < best_acer:
            best_acer = acer
            best_thresh = t

    return best_thresh


def evaluate_all(
    y_true: np.ndarray,
    y_pred_scores: np.ndarray,
    threshold: float = None,
    verbose: bool = True,
) -> dict:
    """
    Compute all anti-spoofing metrics at once.

    Args:
        y_true:          ground truth (0=real, 1=spoof)
        y_pred_scores:   spoof probability from model (after softmax)
        threshold:       decision threshold. If None, finds optimal threshold.
        verbose:         print results table

    Returns:
        dict with keys: auc, eer, eer_threshold, apcer, bpcer, acer, threshold
    """
    if threshold is None:
        threshold = find_best_threshold(y_true, y_pred_scores)

    auc               = compute_auc(y_true, y_pred_scores)
    eer, eer_thresh   = compute_eer(y_true, y_pred_scores)
    apcer             = compute_apcer(y_true, y_pred_scores, threshold)
    bpcer             = compute_bpcer(y_true, y_pred_scores, threshold)
    acer              = compute_acer(apcer, bpcer)

    results = {
        'auc':           auc,
        'eer':           eer,
        'eer_threshold': eer_thresh,
        'apcer':         apcer,
        'bpcer':         bpcer,
        'acer':          acer,
        'threshold':     threshold,
    }

    if verbose:
        print("\n" + "=" * 50)
        print("  ANTI-SPOOFING EVALUATION METRICS")
        print("=" * 50)
        print(f"  AUC    : {auc*100:.2f}%  (higher = better, paper: 99.22%)")
        print(f"  EER    : {eer*100:.2f}%  (lower  = better, threshold: {eer_thresh:.3f})")
        print(f"  APCER  : {apcer*100:.2f}%  (lower  = better, spoof→real errors)")
        print(f"  BPCER  : {bpcer*100:.2f}%  (lower  = better, real→spoof errors)")
        print(f"  ACER   : {acer*100:.2f}%  (lower  = better, paper: 0.81%)")
        print(f"  Thresh : {threshold:.3f}")
        print("=" * 50 + "\n")

    return results



