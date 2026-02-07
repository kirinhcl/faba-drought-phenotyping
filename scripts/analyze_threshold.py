#!/usr/bin/env python3
"""Threshold optimisation for stress detection model.

Sweeps classification thresholds on existing predictions to find optimal
operating points balancing precision, recall, and onset detection accuracy.

No GPU or model needed — reads plant_predictions.csv from presymptomatic analysis.

Usage:
    python scripts/analyze_threshold.py \
        --predictions results/presymptomatic_analysis/plant_predictions.csv \
        --output_dir results/threshold_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

ROUND_TO_DAG = {
    2: 4, 3: 5, 4: 6, 5: 7, 6: 10, 7: 12,
    8: 13, 9: 14, 10: 17, 11: 19, 12: 20, 13: 21,
    14: 24, 15: 27, 16: 28, 17: 29, 18: 31, 19: 33,
    20: 34, 21: 35, 22: 38, 23: 38,
}
DAG_VALUES = [ROUND_TO_DAG[r] for r in range(2, 24)]
T = 22


def load_predictions(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load predictions CSV → (probs, labels, treatments)."""
    df = pd.read_csv(path)
    probs = np.array([[df.iloc[i][f"prob_t{t}"] for t in range(T)] for i in range(len(df))])
    labels = np.array([[df.iloc[i][f"label_t{t}"] for t in range(T)] for i in range(len(df))])
    treatments = df["treatment"].values
    return probs, labels, treatments


def compute_onset_errors(
    probs: np.ndarray,
    labels: np.ndarray,
    treatments: np.ndarray,
    threshold: float,
    onset_range: Tuple[int, int] = (0, 999),
) -> List[int]:
    """Compute onset prediction errors for WHC-30 plants."""
    errors = []
    for i in range(len(probs)):
        if treatments[i] != "WHC-30":
            continue
        true_idx = np.where(labels[i] == 1)[0]
        if len(true_idx) == 0:
            continue
        true_onset = DAG_VALUES[true_idx[0]]
        if not (onset_range[0] <= true_onset <= onset_range[1]):
            continue
        pred_idx = np.where(probs[i] > threshold)[0]
        pred_onset = DAG_VALUES[pred_idx[0]] if len(pred_idx) > 0 else DAG_VALUES[-1]
        errors.append(pred_onset - true_onset)
    return errors


def threshold_sweep(
    probs: np.ndarray,
    labels: np.ndarray,
    treatments: np.ndarray,
) -> List[Dict[str, Any]]:
    """Sweep thresholds and compute all metrics at each."""
    flat_probs = probs.flatten()
    flat_labels = labels.flatten()
    results = []

    for thresh in np.arange(0.05, 0.96, 0.01):
        preds = (flat_probs > thresh).astype(int)
        acc = accuracy_score(flat_labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            flat_labels, preds, average="binary", zero_division=0
        )
        tn = int(((preds == 0) & (flat_labels == 0)).sum())
        fp = int(((preds == 1) & (flat_labels == 0)).sum())
        fn = int(((preds == 0) & (flat_labels == 1)).sum())
        tp = int(((preds == 1) & (flat_labels == 1)).sum())
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        youden = rec + spec - 1

        errors = compute_onset_errors(probs, labels, treatments, thresh)
        mae = np.mean(np.abs(errors)) if errors else 0
        early_rate = np.mean([e < 0 for e in errors]) * 100 if errors else 0
        mean_err = np.mean(errors) if errors else 0
        never_det = sum(1 for e in errors if e == DAG_VALUES[-1] - DAG_VALUES[0]) if errors else 0

        results.append({
            "threshold": round(float(thresh), 2),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "specificity": round(spec, 4),
            "youden_j": round(youden, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "onset_mae": round(float(mae), 2),
            "early_rate": round(float(early_rate), 1),
            "mean_error": round(float(mean_err), 2),
            "never_detected": never_det,
        })
    return results


def find_optimal_thresholds(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Find optimal thresholds under different criteria."""
    best = {}

    # Best F1
    f1_sorted = sorted(results, key=lambda r: -r["f1"])
    best["best_f1"] = f1_sorted[0]

    # Best Youden's J (sensitivity + specificity - 1)
    youden_sorted = sorted(results, key=lambda r: -r["youden_j"])
    best["best_youden"] = youden_sorted[0]

    # Precision ≥ 0.70 with highest recall
    high_prec = [r for r in results if r["precision"] >= 0.70]
    if high_prec:
        best["prec_ge_70"] = sorted(high_prec, key=lambda r: -r["recall"])[0]

    # Precision ≥ 0.80 with highest recall
    high_prec80 = [r for r in results if r["precision"] >= 0.80]
    if high_prec80:
        best["prec_ge_80"] = sorted(high_prec80, key=lambda r: -r["recall"])[0]

    # Lowest onset MAE
    mae_sorted = sorted(results, key=lambda r: r["onset_mae"])
    best["lowest_mae"] = mae_sorted[0]

    # Balanced: F1 × (1 - onset_mae/38)  (penalise high MAE)
    for r in results:
        r["_balanced"] = r["f1"] * (1 - r["onset_mae"] / 38.0)
    balanced_sorted = sorted(results, key=lambda r: -r["_balanced"])
    best["balanced"] = balanced_sorted[0]

    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold optimisation")
    parser.add_argument("--predictions", type=str,
                        default="results/presymptomatic_analysis/plant_predictions.csv")
    parser.add_argument("--output_dir", type=str,
                        default="results/threshold_analysis")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading predictions...")
    probs, labels, treatments = load_predictions(Path(args.predictions))
    print(f"  {len(probs)} plants, {T} timesteps")
    print(f"  Positive: {int(labels.sum())} ({labels.mean()*100:.1f}%)")

    # AUC (threshold-independent)
    flat_probs = probs.flatten()
    flat_labels = labels.flatten()
    auc = roc_auc_score(flat_labels, flat_probs)
    print(f"  AUC: {auc:.4f}")

    # ---- Full sweep ----
    print("\nRunning threshold sweep (0.05 to 0.95, step 0.01)...")
    results = threshold_sweep(probs, labels, treatments)

    # Print summary table (every 5%)
    print(f"\n{'=' * 120}")
    print("THRESHOLD SWEEP RESULTS (every 5%)")
    print(f"{'=' * 120}")
    print(f"  {'Thresh':>6} | {'Acc':>6} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | {'Spec':>6} | "
          f"{'Youden':>6} | {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} | {'MAE':>5} | {'Early%':>6} | {'MeanErr':>7}")
    print(f"  {'-' * 115}")

    for r in results:
        if r["threshold"] * 100 % 5 > 0.5:
            continue
        tag = " ◄ default" if abs(r["threshold"] - 0.50) < 0.01 else ""
        print(f"  {r['threshold']:>6.2f} | {r['accuracy']:>6.3f} | {r['precision']:>6.3f} | "
              f"{r['recall']:>6.3f} | {r['f1']:>6.3f} | {r['specificity']:>6.3f} | "
              f"{r['youden_j']:>6.3f} | "
              f"{r['tp']:>5} {r['fp']:>5} {r['fn']:>5} {r['tn']:>5} | "
              f"{r['onset_mae']:>5.1f} | {r['early_rate']:>5.1f}% | {r['mean_error']:>+6.1f}d{tag}")

    # ---- Optimal thresholds ----
    optimal = find_optimal_thresholds(results)

    print(f"\n{'=' * 120}")
    print("OPTIMAL THRESHOLDS")
    print(f"{'=' * 120}")
    for name, r in optimal.items():
        print(f"\n  {name}:")
        print(f"    Threshold: {r['threshold']:.2f}")
        print(f"    Prec={r['precision']:.3f}  Rec={r['recall']:.3f}  F1={r['f1']:.3f}  Spec={r['specificity']:.3f}")
        print(f"    Onset MAE={r['onset_mae']:.1f}d  Early%={r['early_rate']:.1f}%  MeanErr={r['mean_error']:+.1f}d")

    # ---- Per-category at optimal thresholds ----
    print(f"\n{'=' * 120}")
    print("PER-CATEGORY ONSET METRICS AT KEY THRESHOLDS")
    print(f"{'=' * 120}")

    key_thresholds = {
        "default (0.50)": 0.50,
        f"best-F1 ({optimal['best_f1']['threshold']:.2f})": optimal["best_f1"]["threshold"],
    }
    if "prec_ge_70" in optimal:
        key_thresholds[f"prec≥0.70 ({optimal['prec_ge_70']['threshold']:.2f})"] = optimal["prec_ge_70"]["threshold"]
    if "prec_ge_80" in optimal:
        key_thresholds[f"prec≥0.80 ({optimal['prec_ge_80']['threshold']:.2f})"] = optimal["prec_ge_80"]["threshold"]

    for name, thresh in key_thresholds.items():
        print(f"\n  === {name} ===")
        print(f"  {'Category':>10} | {'N':>3} | {'MAE':>6} | {'Early%':>6} | {'MeanErr':>8} | {'NeverDet':>8}")
        print(f"  {'-' * 55}")

        for cat_label, rng in [("Early", (0, 14)), ("Mid", (15, 23)), ("Late", (24, 999)), ("ALL", (0, 999))]:
            errors = compute_onset_errors(probs, labels, treatments, thresh, rng)
            if errors:
                mae = np.mean(np.abs(errors))
                early = np.mean([e < 0 for e in errors]) * 100
                mean_err = np.mean(errors)
                never = sum(1 for i in range(len(probs))
                            if treatments[i] == "WHC-30"
                            and len(np.where(labels[i] == 1)[0]) > 0
                            and rng[0] <= DAG_VALUES[np.where(labels[i] == 1)[0][0]] <= rng[1]
                            and len(np.where(probs[i] > thresh)[0]) == 0)
                print(f"  {cat_label:>10} | {len(errors):>3} | {mae:>6.1f} | {early:>5.1f}% | {mean_err:>+7.1f}d | {never:>8}")

    # ---- Recommendation ----
    rec_thresh = optimal["best_f1"]["threshold"]
    rec = optimal["best_f1"]
    print(f"\n{'=' * 120}")
    print("RECOMMENDATION")
    print(f"{'=' * 120}")
    print(f"""
  For balanced performance, use threshold = {rec_thresh:.2f}
    Precision: {rec['precision']:.3f}  (vs 0.549 at 0.50)
    Recall:    {rec['recall']:.3f}  (vs 0.867 at 0.50)
    F1:        {rec['f1']:.3f}  (vs 0.621 at 0.50)
    Onset MAE: {rec['onset_mae']:.1f}d  (vs 11.4d at 0.50)
""")

    if "prec_ge_70" in optimal:
        p70 = optimal["prec_ge_70"]
        print(f"  For high-precision applications, use threshold = {p70['threshold']:.2f}")
        print(f"    Precision: {p70['precision']:.3f}  Recall: {p70['recall']:.3f}  F1: {p70['f1']:.3f}  MAE: {p70['onset_mae']:.1f}d")

    # Save JSON
    output = {
        "auc": round(auc, 4),
        "sweep": results,
        "optimal": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in optimal.items()},
    }
    json_path = output_dir / "threshold_analysis.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
