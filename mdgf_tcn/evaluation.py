"""Candidate-region metrics with the archived validation threshold rule."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def validation_threshold(labels, scores):
    """Maximize MCC, breaking ties by F1 and then the smaller threshold."""
    order = np.argsort(-scores, kind="stable")
    ordered_scores, ordered_labels = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered_scores)), len(scores) - 1]
    tp = np.cumsum(ordered_labels)[ends].astype(float)
    fp = ends + 1 - tp
    fn = labels.sum() - tp
    tn = len(labels) - tp - fp - fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, denominator, out=np.zeros_like(tp), where=denominator > 0)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    thresholds = ordered_scores[ends]
    return float(thresholds[np.lexsort((thresholds, -f1, -mcc))[0]])


def metrics(labels, scores, threshold=None):
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    if labels.size != scores.size or set(np.unique(labels)) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Metrics require aligned finite scores and both binary classes")
    if threshold is None:
        threshold = validation_threshold(labels, scores)
    elif not np.isfinite(threshold):
        raise ValueError("The decision threshold must be finite")
    tn, fp, fn, tp = map(int, confusion_matrix(labels, scores >= threshold, labels=[0, 1]).ravel())
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "N": len(labels),
        "threshold": float(threshold),
        "Accuracy": (tn + tp) / len(labels),
        "Precision": tp / max(tp + fp, 1),
        "Recall": tp / max(tp + fn, 1),
        "F1": 2 * tp / max(2 * tp + fp + fn, 1),
        "MCC": (tp * tn - fp * fn) / denominator if denominator else 0.0,
        "AUROC": float(roc_auc_score(labels, scores)),
        "AP": float(average_precision_score(labels, scores)),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def write_predictions(path: Path, indices, labels, scores, threshold):
    np.savez_compressed(
        path, indices=indices, y_true=labels[indices], scores=np.asarray(scores, dtype=np.float64),
        threshold=np.array([threshold]), y_pred=(np.asarray(scores) >= threshold).astype(np.int64),
    )


def score_archive(path: Path):
    """Recompute the metrics without changing the saved scores or threshold."""
    with np.load(path, allow_pickle=False) as archive:
        return metrics(archive["y_true"], archive["scores"], float(archive["threshold"].item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score an archived validation or test prediction file.")
    parser.add_argument("predictions", type=Path, help="NPZ file with y_true, scores and threshold")
    print(json.dumps(score_archive(parser.parse_args().predictions), indent=2))
