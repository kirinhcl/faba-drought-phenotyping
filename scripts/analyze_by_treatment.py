#!/usr/bin/env python3
"""Per-treatment evaluation breakdown.

Splits per-timestep metrics into:
  - WHC-80 (control): should be all label=0, check false positive rate
  - WHC-30 pre-onset: label=0, check false positive rate
  - WHC-30 post-onset: label=1, check true positive rate

Usage:
    python scripts/analyze_by_treatment.py \
        --predictions results/presymptomatic_v3/plant_predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROUND_TO_DAG = {
    2: 4, 3: 5, 4: 6, 5: 7, 6: 10, 7: 12,
    8: 13, 9: 14, 10: 17, 11: 19, 12: 20, 13: 21,
    14: 24, 15: 27, 16: 28, 17: 29, 18: 31, 19: 33,
    20: 34, 21: 35, 22: 38, 23: 38,
}
DAG_VALUES = [ROUND_TO_DAG[r] for r in range(2, 24)]
T = 22


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)

    probs = np.array([[df.iloc[i][f"prob_t{t}"] for t in range(T)] for i in range(len(df))])
    labels = np.array([[df.iloc[i][f"label_t{t}"] for t in range(T)] for i in range(len(df))])
    treatments = df["treatment"].values

    for thresh in [0.50]:
        preds = (probs > thresh).astype(int)

        print(f"{'=' * 80}")
        print(f"THRESHOLD = {thresh}")
        print(f"{'=' * 80}")

        # --- WHC-80 ---
        ctrl_mask = treatments == "WHC-80"
        ctrl_preds = preds[ctrl_mask]
        ctrl_labels = labels[ctrl_mask]
        ctrl_probs = probs[ctrl_mask]

        ctrl_total = ctrl_preds.size
        ctrl_fp = ctrl_preds.sum()
        ctrl_fp_rate = ctrl_fp / ctrl_total * 100

        print(f"\n  WHC-80 (Control) — {ctrl_mask.sum()} plants, {ctrl_total} timesteps")
        print(f"    All labels = 0 (should never be predicted as stressed)")
        print(f"    False positives: {ctrl_fp} / {ctrl_total} ({ctrl_fp_rate:.1f}%)")
        print(f"    Specificity: {(1 - ctrl_fp / ctrl_total) * 100:.1f}%")
        print(f"    Mean prob: {ctrl_probs.mean():.4f}")
        print(f"    Max prob:  {ctrl_probs.max():.4f}")

        # Per-timepoint FP rate for control
        print(f"\n    Per-timepoint false positive rate:")
        print(f"    {'DAG':>5} | {'Mean prob':>9} | {'FP rate':>7} | {'Bar'}")
        print(f"    {'-' * 45}")
        for t in range(T):
            dag = DAG_VALUES[t]
            t_probs = ctrl_probs[:, t]
            t_preds = ctrl_preds[:, t]
            fp_rate = t_preds.sum() / len(t_preds) * 100
            bar = "█" * int(fp_rate / 2)
            print(f"    {dag:>5} | {t_probs.mean():>9.4f} | {fp_rate:>6.1f}% | {bar}")

        # --- WHC-30 ---
        drgt_mask = treatments == "WHC-30"
        drgt_preds = preds[drgt_mask]
        drgt_labels = labels[drgt_mask]
        drgt_probs = probs[drgt_mask]

        # Split into pre-onset and post-onset
        pre_preds, pre_labels, pre_probs = [], [], []
        post_preds, post_labels, post_probs = [], [], []

        for i in range(drgt_preds.shape[0]):
            for t in range(T):
                if drgt_labels[i, t] == 0:
                    pre_preds.append(drgt_preds[i, t])
                    pre_labels.append(0)
                    pre_probs.append(drgt_probs[i, t])
                else:
                    post_preds.append(drgt_preds[i, t])
                    post_labels.append(1)
                    post_probs.append(drgt_probs[i, t])

        pre_preds = np.array(pre_preds)
        post_preds = np.array(post_preds)
        pre_probs = np.array(pre_probs)
        post_probs = np.array(post_probs)

        pre_fp = pre_preds.sum()
        pre_fp_rate = pre_fp / len(pre_preds) * 100 if len(pre_preds) > 0 else 0
        post_tp = post_preds.sum()
        post_tp_rate = post_tp / len(post_preds) * 100 if len(post_preds) > 0 else 0

        print(f"\n  WHC-30 PRE-ONSET — {len(pre_preds)} timesteps (label=0)")
        print(f"    False positives: {pre_fp} / {len(pre_preds)} ({pre_fp_rate:.1f}%)")
        print(f"    Specificity: {(1 - pre_fp / len(pre_preds)) * 100:.1f}%")
        print(f"    Mean prob: {pre_probs.mean():.4f}")

        print(f"\n  WHC-30 POST-ONSET — {len(post_preds)} timesteps (label=1)")
        print(f"    True positives: {post_tp} / {len(post_preds)} ({post_tp_rate:.1f}%)")
        print(f"    Recall: {post_tp_rate:.1f}%")
        print(f"    Mean prob: {post_probs.mean():.4f}")

        # --- Per-plant accuracy for WHC-80 ---
        print(f"\n  {'=' * 70}")
        print(f"  WHC-80 PER-PLANT: any false positive?")
        print(f"  {'=' * 70}")
        ctrl_df = df[df["treatment"] == "WHC-80"]
        n_clean = 0
        n_dirty = 0
        dirty_plants = []
        for idx in range(len(ctrl_df)):
            row = ctrl_df.iloc[idx]
            plant_probs = np.array([row[f"prob_t{t}"] for t in range(T)])
            plant_preds = (plant_probs > thresh).astype(int)
            fp_count = plant_preds.sum()
            if fp_count > 0:
                n_dirty += 1
                dirty_plants.append({
                    "plant_id": row["plant_id"],
                    "accession": row["accession"],
                    "fp_count": int(fp_count),
                    "max_prob": float(plant_probs.max()),
                    "fp_timesteps": [DAG_VALUES[t] for t in range(T) if plant_preds[t] == 1],
                })
            else:
                n_clean += 1

        print(f"    Perfect (0 FP): {n_clean} / {n_clean + n_dirty} ({n_clean/(n_clean+n_dirty)*100:.1f}%)")
        print(f"    Has FP:         {n_dirty} / {n_clean + n_dirty} ({n_dirty/(n_clean+n_dirty)*100:.1f}%)")

        if dirty_plants:
            print(f"\n    Plants with false positives:")
            print(f"    {'Plant ID':>20} | {'Accession':>20} | {'FP':>3} | {'Max prob':>8} | FP at DAGs")
            print(f"    {'-' * 75}")
            for p in sorted(dirty_plants, key=lambda x: -x["fp_count"]):
                dags = ", ".join(str(d) for d in p["fp_timesteps"])
                print(f"    {p['plant_id']:>20} | {p['accession']:>20} | {p['fp_count']:>3} | {p['max_prob']:>8.4f} | {dags}")

        # --- Summary ---
        print(f"\n  {'=' * 70}")
        print(f"  SUMMARY")
        print(f"  {'=' * 70}")
        print(f"    WHC-80 specificity:        {(1 - ctrl_fp / ctrl_total) * 100:.1f}% (FP={ctrl_fp}/{ctrl_total})")
        print(f"    WHC-30 pre-onset spec:     {(1 - pre_fp / len(pre_preds)) * 100:.1f}% (FP={pre_fp}/{len(pre_preds)})")
        print(f"    WHC-30 post-onset recall:  {post_tp_rate:.1f}% (TP={post_tp}/{len(post_preds)})")
        print(f"    WHC-80 plant-level clean:  {n_clean}/{n_clean + n_dirty} ({n_clean/(n_clean+n_dirty)*100:.1f}%)")


if __name__ == "__main__":
    main()
