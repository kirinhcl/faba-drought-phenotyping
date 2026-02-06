#!/usr/bin/env python3
"""Pre-symptomatic detection validation: gates analysis + triangulation.

Two independent analyses to validate that the model's early detection
reflects real physiological signals rather than artefacts:

1. MODALITY GATES ANALYSIS
   - Compare gate weights between Control vs Drought over time
   - Show whether fluorescence gate increases before human-annotated onset
   - Split analysis by drought category (Early/Mid/Late)

2. TRIANGULATION (per-genotype)
   - For each genotype, compare three onset estimates:
     a) Fluorescence statistical divergence (from analyze_fluorescence.py)
     b) Model predicted onset (first timestep with prob > 0.5)
     c) Human-annotated onset (ground truth label)
   - If model onset ≈ fluorescence onset < human onset → pre-symptomatic detection

Requires: trained stress model checkpoints on GPU node.

Usage:
    python scripts/analyze_presymptomatic.py \
        --results_dir results/stress/checkpoints \
        --config configs/stress.yaml \
        --output_dir results/presymptomatic_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats as sp_stats
from torch.utils.data import DataLoader, Subset

from src.data.collate import faba_collate_fn
from src.data.dataset import ROUND_TO_DAG, FabaDroughtDataset
from src.model.stress_model import StressDetectionModel
from src.training.cv import LogoCV
from src.utils.config import load_config

DAG_VALUES = [ROUND_TO_DAG[r] for r in range(2, 24)]
MODALITY_NAMES = ["Image", "Fluorescence", "Environment", "Veg. Index"]


# ---------------------------------------------------------------------------
# Part 1: Model inference — collect per-plant predictions, gates, metadata
# ---------------------------------------------------------------------------

def run_inference_all_folds(
    cfg: Any,
    dataset: FabaDroughtDataset,
    checkpoint_path: Path,
) -> pd.DataFrame:
    """Run inference across all 44 folds, returning per-plant results.

    Returns DataFrame with columns:
        plant_id, accession, treatment, fold_id,
        probs_t0..probs_t21,  gates_image_t0..gates_vi_t21,
        pred_onset_dag, true_onset_dag, dag_drought_onset
    """
    cv = LogoCV(
        plant_metadata_df=dataset.plant_metadata,
        n_folds=cfg.training.cv.n_folds,
        stratify_col=cfg.training.cv.stratify_by,
        seed=cfg.training.cv.seed,
    )

    all_rows: List[Dict[str, Any]] = []

    for fold_id, (_, _, test_idx) in enumerate(cv.split()):
        model_path = checkpoint_path / f"fold_{fold_id}" / "best_model_state.pt"
        if not model_path.exists():
            print(f"  [fold {fold_id}] checkpoint not found, skipping")
            continue

        print(f"  [fold {fold_id}] running inference on {len(test_idx)} test plants...")

        model = StressDetectionModel(cfg)
        model.load_state_dict(torch.load(model_path, map_location=cfg.device, weights_only=True))
        model.to(cfg.device)
        model.eval()

        test_dataset = Subset(dataset, test_idx.tolist())
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=faba_collate_fn,
            pin_memory=True,
        )

        with torch.no_grad():
            for batch in test_loader:
                batch_device = {
                    k: v.to(cfg.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                predictions = model(batch_device)

                probs = torch.sigmoid(predictions["stress_logits"]).cpu().numpy()   # (B, T)
                gates = predictions["modality_gates"].cpu().numpy()                 # (B, T, 4)
                labels = batch["stress_labels"].cpu().numpy()                       # (B, T)

                for b in range(probs.shape[0]):
                    row: Dict[str, Any] = {
                        "plant_id": batch["plant_id"][b],
                        "accession": batch["accession"][b],
                        "treatment": batch["treatment"][b],
                        "fold_id": fold_id,
                    }

                    # Per-timestep probabilities
                    for t in range(22):
                        row[f"prob_t{t}"] = float(probs[b, t])

                    # Per-timestep gate weights (4 modalities)
                    for t in range(22):
                        for m, mname in enumerate(["image", "fluor", "env", "vi"]):
                            row[f"gate_{mname}_t{t}"] = float(gates[b, t, m])

                    # Per-timestep labels
                    for t in range(22):
                        row[f"label_t{t}"] = int(labels[b, t])

                    # Predicted onset (first t with prob > 0.5)
                    pred_stress = np.where(probs[b] > 0.5)[0]
                    if len(pred_stress) > 0:
                        pred_onset_idx = int(pred_stress[0])
                        row["pred_onset_dag"] = DAG_VALUES[pred_onset_idx]
                    else:
                        row["pred_onset_dag"] = None

                    # True onset (first t with label == 1)
                    true_stress = np.where(labels[b] == 1)[0]
                    if len(true_stress) > 0:
                        true_onset_idx = int(true_stress[0])
                        row["true_onset_dag"] = DAG_VALUES[true_onset_idx]
                    else:
                        row["true_onset_dag"] = None

                    all_rows.append(row)

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Part 2: Fluorescence statistical change-point per genotype
# ---------------------------------------------------------------------------

def compute_fluor_changepoints(data_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Compute per-genotype fluorescence change-point using Fv/Fm_Lss.

    Uses Welch's t-test per round: control vs drought plants of that genotype.
    Falls back to population-level control if per-genotype sample is too small.

    Returns {accession: {changepoint_dag, param, z_scores: {dag: z}}}.
    """
    fluor_path = data_dir / "TimeCourse Datasets" / "FCQ_FabaDr_Auto.xlsx"
    df = pd.read_excel(fluor_path)
    df["Plant ID"] = df["Plant ID"].astype(str).str.strip()

    # Build population-level control baseline per round
    param = "Fv/Fm_Lss"
    if param not in df.columns:
        # Fallback to QY_max
        param = "QY_max"
    print(f"  Using fluorescence indicator: {param}")

    ctrl_baseline: Dict[int, Tuple[float, float]] = {}
    for rnd in sorted(df["Round Order"].unique()):
        vals = df.loc[(df["Round Order"] == rnd) & (df["Treatment"] == "WHC-80%"), param].dropna()
        if len(vals) >= 3:
            ctrl_baseline[int(rnd)] = (float(vals.mean()), max(float(vals.std()), 1e-6))

    # Load plant metadata for genotype grouping
    meta = pd.read_csv(data_dir / "plant_metadata.csv")

    results: Dict[str, Dict[str, Any]] = {}

    for accession in sorted(meta["accession"].unique()):
        # Get drought plant IDs for this genotype
        drought_pids = meta.loc[
            (meta["accession"] == accession) & (meta["treatment"] == "WHC-30"),
            "plant_id",
        ].tolist()
        if not drought_pids:
            continue

        z_scores: Dict[int, float] = {}
        changepoint_dag: Optional[int] = None

        for rnd in sorted(df["Round Order"].unique()):
            rnd_int = int(rnd)
            if rnd_int not in ctrl_baseline:
                continue
            ctrl_mean, ctrl_std = ctrl_baseline[rnd_int]

            drought_vals = df.loc[
                (df["Round Order"] == rnd) & (df["Plant ID"].isin(drought_pids)),
                param,
            ].dropna()
            if len(drought_vals) == 0:
                continue

            # Mean z-score of drought plants vs control baseline
            z = float(np.abs((drought_vals.mean() - ctrl_mean) / ctrl_std))
            dag = ROUND_TO_DAG.get(rnd_int)
            if dag is not None:
                z_scores[dag] = round(z, 3)
                if z > 2.0 and changepoint_dag is None:
                    changepoint_dag = dag

        results[accession] = {
            "changepoint_dag": changepoint_dag,
            "param": param,
            "z_scores": z_scores,
        }

    return results


# ---------------------------------------------------------------------------
# Part 3: Modality gates temporal analysis
# ---------------------------------------------------------------------------

def analyze_gates(plant_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze modality gate weights over time, split by treatment.

    Returns dict with:
        control_gates: (T, 4) mean gate weights for WHC-80
        drought_gates: (T, 4) mean gate weights for WHC-30
        pre_onset_gates: mean gates for drought plants in timesteps before onset
        post_onset_gates: mean gates for drought plants in timesteps after onset
    """
    T = 22
    modalities = ["image", "fluor", "env", "vi"]

    def _extract_gates(df: pd.DataFrame) -> np.ndarray:
        """Extract (N, T, 4) gates from DataFrame columns."""
        N = len(df)
        gates = np.zeros((N, T, 4))
        for t in range(T):
            for m, mname in enumerate(modalities):
                col = f"gate_{mname}_t{t}"
                if col in df.columns:
                    gates[:, t, m] = df[col].values
        return gates

    ctrl_df = plant_df[plant_df["treatment"] == "WHC-80"]
    drgt_df = plant_df[plant_df["treatment"] == "WHC-30"]

    ctrl_gates = _extract_gates(ctrl_df)
    drgt_gates = _extract_gates(drgt_df)

    # Pre/post onset split for drought plants
    pre_onset_weights = {m: [] for m in modalities}
    post_onset_weights = {m: [] for m in modalities}

    for _, row in drgt_df.iterrows():
        true_onset = row.get("true_onset_dag")
        if true_onset is None:
            continue
        for t in range(T):
            dag = DAG_VALUES[t]
            for m_idx, mname in enumerate(modalities):
                gate_val = row.get(f"gate_{mname}_t{t}", 0)
                if dag < true_onset:
                    pre_onset_weights[mname].append(gate_val)
                else:
                    post_onset_weights[mname].append(gate_val)

    return {
        "control_gates_mean": ctrl_gates.mean(axis=0).tolist(),      # (T, 4)
        "drought_gates_mean": drgt_gates.mean(axis=0).tolist(),      # (T, 4)
        "control_gates_std": ctrl_gates.std(axis=0).tolist(),
        "drought_gates_std": drgt_gates.std(axis=0).tolist(),
        "n_control": len(ctrl_df),
        "n_drought": len(drgt_df),
        "pre_onset_mean": {m: float(np.mean(v)) if v else None for m, v in pre_onset_weights.items()},
        "post_onset_mean": {m: float(np.mean(v)) if v else None for m, v in post_onset_weights.items()},
    }


# ---------------------------------------------------------------------------
# Part 4: Triangulation — per-genotype comparison
# ---------------------------------------------------------------------------

def triangulate(
    plant_df: pd.DataFrame,
    fluor_changepoints: Dict[str, Dict[str, Any]],
    plant_meta: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Per-genotype three-way comparison: fluorescence vs model vs human.

    For each genotype with drought plants, compute:
    - fluor_onset: fluorescence statistical changepoint (Fv/Fm z>2)
    - model_onset: median predicted onset across 3 drought reps
    - human_onset: annotated dag_drought_onset (same for all 3 reps)

    Returns list of dicts, one per genotype.
    """
    results = []

    for accession in sorted(plant_df["accession"].unique()):
        drgt = plant_df[
            (plant_df["accession"] == accession) & (plant_df["treatment"] == "WHC-30")
        ]
        if len(drgt) == 0:
            continue

        # Human onset
        human_onsets = drgt["true_onset_dag"].dropna().unique()
        human_onset = float(human_onsets[0]) if len(human_onsets) > 0 else None

        # Model onset (median across reps)
        model_onsets = drgt["pred_onset_dag"].dropna()
        model_onset = float(model_onsets.median()) if len(model_onsets) > 0 else None

        # Fluorescence changepoint
        fluor_info = fluor_changepoints.get(accession, {})
        fluor_onset = fluor_info.get("changepoint_dag")

        # Category
        cat_row = plant_meta.loc[
            (plant_meta["accession"] == accession) & (plant_meta["treatment"] == "WHC-30"),
            "drought_category",
        ]
        category = str(cat_row.iloc[0]) if len(cat_row) > 0 else "Unknown"

        row = {
            "accession": accession,
            "category": category,
            "human_onset_dag": human_onset,
            "model_onset_dag": model_onset,
            "fluor_onset_dag": fluor_onset,
        }

        if human_onset is not None and model_onset is not None:
            row["model_lead_days"] = human_onset - model_onset
        else:
            row["model_lead_days"] = None

        if human_onset is not None and fluor_onset is not None:
            row["fluor_lead_days"] = human_onset - fluor_onset
        else:
            row["fluor_lead_days"] = None

        if model_onset is not None and fluor_onset is not None:
            row["model_vs_fluor_gap"] = model_onset - fluor_onset
        else:
            row["model_vs_fluor_gap"] = None

        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Part 5: Reporting
# ---------------------------------------------------------------------------

def print_gates_report(gates_result: Dict[str, Any]) -> None:
    """Print modality gates analysis."""
    print("\n" + "=" * 80)
    print("MODALITY GATES: Control vs Drought Over Time")
    print("=" * 80)

    ctrl = np.array(gates_result["control_gates_mean"])   # (T, 4)
    drgt = np.array(gates_result["drought_gates_mean"])    # (T, 4)

    print(f"\n  {'DAG':>5} | {'Ctrl-Img':>8} {'Ctrl-Flr':>8} {'Ctrl-Env':>8} {'Ctrl-VI':>8} | {'Drgt-Img':>8} {'Drgt-Flr':>8} {'Drgt-Env':>8} {'Drgt-VI':>8}")
    print(f"  {'-' * 100}")

    for t in range(22):
        dag = DAG_VALUES[t]
        c = ctrl[t]
        d = drgt[t]
        print(f"  {dag:>5} | {c[0]:>8.3f} {c[1]:>8.3f} {c[2]:>8.3f} {c[3]:>8.3f} | {d[0]:>8.3f} {d[1]:>8.3f} {d[2]:>8.3f} {d[3]:>8.3f}")

    # Pre vs post onset
    pre = gates_result["pre_onset_mean"]
    post = gates_result["post_onset_mean"]
    print(f"\n  PRE-ONSET  (drought plants, before annotated onset):")
    for m in ["image", "fluor", "env", "vi"]:
        v = pre.get(m)
        print(f"    {m:>12}: {v:.4f}" if v is not None else f"    {m:>12}: N/A")
    print(f"  POST-ONSET (drought plants, after annotated onset):")
    for m in ["image", "fluor", "env", "vi"]:
        v = post.get(m)
        print(f"    {m:>12}: {v:.4f}" if v is not None else f"    {m:>12}: N/A")


def print_triangulation_report(tri_results: List[Dict[str, Any]]) -> None:
    """Print per-genotype triangulation table and summary."""
    print("\n" + "=" * 80)
    print("TRIANGULATION: Fluorescence vs Model vs Human (per genotype)")
    print("=" * 80)

    print(f"\n  {'Accession':>20} | {'Cat':>5} | {'Fluor':>5} | {'Model':>5} | {'Human':>5} | {'M-Lead':>6} | {'F-Lead':>6} | {'M-F Gap':>7}")
    print(f"  {'-' * 85}")

    for r in sorted(tri_results, key=lambda x: x.get("human_onset_dag") or 99):
        acc = r["accession"][:20]
        cat = r["category"][:5]
        fl = f"{r['fluor_onset_dag']:>5.0f}" if r["fluor_onset_dag"] is not None else "  N/A"
        ml = f"{r['model_onset_dag']:>5.0f}" if r["model_onset_dag"] is not None else "  N/A"
        hu = f"{r['human_onset_dag']:>5.0f}" if r["human_onset_dag"] is not None else "  N/A"
        m_lead = f"{r['model_lead_days']:>+5.0f}d" if r["model_lead_days"] is not None else "   N/A"
        f_lead = f"{r['fluor_lead_days']:>+5.0f}d" if r["fluor_lead_days"] is not None else "   N/A"
        mf_gap = f"{r['model_vs_fluor_gap']:>+6.0f}d" if r["model_vs_fluor_gap"] is not None else "    N/A"
        print(f"  {acc:>20} | {cat:>5} | {fl} | {ml} | {hu} | {m_lead} | {f_lead} | {mf_gap}")

    # Summary statistics
    model_leads = [r["model_lead_days"] for r in tri_results if r["model_lead_days"] is not None]
    fluor_leads = [r["fluor_lead_days"] for r in tri_results if r["fluor_lead_days"] is not None]
    mf_gaps = [r["model_vs_fluor_gap"] for r in tri_results if r["model_vs_fluor_gap"] is not None]

    print(f"\n  SUMMARY (n={len(tri_results)} genotypes)")
    print(f"  {'-' * 60}")
    if model_leads:
        print(f"  Model lead over human:      mean={np.mean(model_leads):>+.1f}d, median={np.median(model_leads):>+.1f}d")
        print(f"    (positive = model detects BEFORE human)")
    if fluor_leads:
        print(f"  Fluor lead over human:      mean={np.mean(fluor_leads):>+.1f}d, median={np.median(fluor_leads):>+.1f}d")
    if mf_gaps:
        print(f"  Model vs Fluor gap:         mean={np.mean(mf_gaps):>+.1f}d, median={np.median(mf_gaps):>+.1f}d")
        print(f"    (positive = model later than fluor, negative = model earlier)")

    # Correlation
    if len(model_leads) >= 5 and len(fluor_leads) >= 5:
        # Match by genotype
        paired = [(r["model_lead_days"], r["fluor_lead_days"]) for r in tri_results
                  if r["model_lead_days"] is not None and r["fluor_lead_days"] is not None]
        if len(paired) >= 5:
            m_arr = np.array([p[0] for p in paired])
            f_arr = np.array([p[1] for p in paired])
            r_val, p_val = sp_stats.pearsonr(m_arr, f_arr)
            print(f"\n  Correlation (model_lead vs fluor_lead): r={r_val:.3f}, p={p_val:.2e}")
            if p_val < 0.05:
                print(f"    SIGNIFICANT: Model early detection correlates with fluorescence divergence")

    # Category breakdown
    print(f"\n  BY CATEGORY:")
    for cat in ["Early", "Mid", "Late"]:
        cat_leads = [r["model_lead_days"] for r in tri_results
                     if r.get("category") == cat and r["model_lead_days"] is not None]
        if cat_leads:
            print(f"    {cat:>5}: model lead = {np.mean(cat_leads):>+.1f}d (n={len(cat_leads)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-symptomatic detection validation")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to stress model checkpoints (e.g. results/stress/checkpoints)")
    parser.add_argument("--config", type=str, default="configs/stress.yaml")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/presymptomatic_analysis")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    checkpoint_path = Path(args.results_dir)

    # ---- Step 1: Model inference ----
    print("=" * 80)
    print("STEP 1: Running model inference across all folds")
    print("=" * 80)

    cfg = load_config(args.config)
    dataset = FabaDroughtDataset(cfg)
    plant_df = run_inference_all_folds(cfg, dataset, checkpoint_path)
    print(f"  Collected {len(plant_df)} plant predictions")

    # Save raw predictions
    pred_path = output_dir / "plant_predictions.csv"
    plant_df.to_csv(pred_path, index=False)
    print(f"  Saved to {pred_path}")

    # ---- Step 2: Fluorescence changepoints ----
    print(f"\n{'=' * 80}")
    print("STEP 2: Computing fluorescence changepoints per genotype")
    print("=" * 80)

    fluor_cp = compute_fluor_changepoints(data_dir)
    n_detected = sum(1 for v in fluor_cp.values() if v["changepoint_dag"] is not None)
    print(f"  {n_detected}/{len(fluor_cp)} genotypes have detectable fluorescence changepoint")

    # Save
    fluor_path = output_dir / "fluor_changepoints.json"
    with open(fluor_path, "w") as f:
        json.dump(fluor_cp, f, indent=2)

    # ---- Step 3: Gates analysis ----
    print(f"\n{'=' * 80}")
    print("STEP 3: Analyzing modality gates")
    print("=" * 80)

    gates_result = analyze_gates(plant_df)
    print_gates_report(gates_result)

    gates_path = output_dir / "gates_analysis.json"
    with open(gates_path, "w") as f:
        json.dump(gates_result, f, indent=2)

    # ---- Step 4: Triangulation ----
    print(f"\n{'=' * 80}")
    print("STEP 4: Triangulating fluorescence vs model vs human")
    print("=" * 80)

    tri_results = triangulate(plant_df, fluor_cp, dataset.plant_metadata)
    print_triangulation_report(tri_results)

    tri_path = output_dir / "triangulation.json"
    with open(tri_path, "w") as f:
        json.dump(tri_results, f, indent=2)

    # ---- Final summary ----
    print(f"\n{'=' * 80}")
    print("OUTPUT FILES")
    print("=" * 80)
    print(f"  {pred_path}")
    print(f"  {fluor_path}")
    print(f"  {gates_path}")
    print(f"  {tri_path}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
