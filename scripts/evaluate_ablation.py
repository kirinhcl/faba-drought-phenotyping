#!/usr/bin/env python3
"""Evaluate ablation experiments with fold-level confidence intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Subset

from src.data.collate import faba_collate_fn
from src.data.dataset import FabaDroughtDataset, ROUND_TO_DAG
from src.model.stress_model import StressDetectionModel
from src.training.cv import LogoCV
from src.utils.config import load_config


def compute_onset_metrics(
    probs: np.ndarray,  # (N, T) probabilities
    labels: np.ndarray,  # (N, T) binary labels
    masks: np.ndarray,  # (N, T) valid masks
    treatments: List[str],  # List of treatment strings
) -> Dict[str, float]:
    """Compute onset detection metrics for WHC-30 plants."""
    _ = masks
    results = []

    for i in range(len(probs)):
        if treatments[i] == "WHC-80":
            continue

        if labels[i].sum() == 0:
            continue

        stress_indices = np.where(labels[i] == 1)[0]
        if len(stress_indices) == 0:
            continue
        true_onset_idx = stress_indices[0]
        true_onset_dag = ROUND_TO_DAG[true_onset_idx + 2]

        pred_stress_indices = np.where(probs[i] > 0.5)[0]
        if len(pred_stress_indices) == 0:
            pred_onset_idx = len(probs[i]) - 1
        else:
            pred_onset_idx = pred_stress_indices[0]
        pred_onset_dag = ROUND_TO_DAG[pred_onset_idx + 2]

        error = pred_onset_dag - true_onset_dag

        results.append({
            "true_onset_dag": true_onset_dag,
            "pred_onset_dag": pred_onset_dag,
            "error": error,
        })

    if len(results) == 0:
        return {
            "onset_mae": float("nan"),
            "mean_onset_error": float("nan"),
            "n_plants": 0,
        }

    errors = np.array([r["error"] for r in results], dtype=float)
    return {
        "onset_mae": float(np.mean(np.abs(errors))),
        "mean_onset_error": float(np.mean(errors)),
        "n_plants": len(results),
    }


def evaluate_fold(
    cfg: Any,
    dataset: FabaDroughtDataset,
    fold_id: int,
    test_indices: np.ndarray,
    checkpoint_root: Path,
) -> Dict[str, Any]:
    """Evaluate a single fold."""
    checkpoint_file = checkpoint_root / f"fold_{fold_id}" / "best_model_state.pt"
    model = StressDetectionModel(cfg)
    model.load_state_dict(torch.load(
        checkpoint_file,
        map_location=cfg.device,
        weights_only=True,
    ))
    model.to(cfg.device)
    model.eval()

    test_dataset = Subset(dataset, test_indices.tolist())
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=faba_collate_fn,
        pin_memory=True,
    )

    all_probs = []
    all_labels = []
    all_masks = []
    all_treatments = []

    with torch.no_grad():
        for batch in test_loader:
            batch_device = {
                k: v.to(cfg.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            predictions = model(batch_device)
            logits = predictions["stress_logits"]

            probs = torch.sigmoid(logits).cpu().numpy()
            labels = batch["stress_labels"].cpu().numpy()
            masks = batch["stress_mask"].cpu().numpy()

            all_probs.append(probs)
            all_labels.append(labels)
            all_masks.append(masks)
            all_treatments.extend(batch["treatment"])

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    valid_probs = all_probs[all_masks]
    valid_labels = all_labels[all_masks]
    valid_preds = (valid_probs > 0.5).astype(int)

    accuracy = accuracy_score(valid_labels, valid_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        valid_labels,
        valid_preds,
        average="binary",
        zero_division=0,  # pyright: ignore[reportArgumentType]
    )
    try:
        auc = roc_auc_score(valid_labels, valid_probs)
    except ValueError:
        auc = float("nan")

    onset_metrics = compute_onset_metrics(all_probs, all_labels, all_masks, all_treatments)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        **onset_metrics,
    }


def bootstrap_ci(values: Iterable[float], n_bootstrap: int = 10000, ci: float = 0.95) -> Dict[str, float]:
    """Compute bootstrap confidence interval of the mean."""
    values_array = np.array(list(values), dtype=float)
    values_array = values_array[~np.isnan(values_array)]
    if values_array.size == 0:
        return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}

    n = len(values_array)
    bootstrap_means = np.array([
        np.mean(np.random.choice(values_array, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    return {"mean": float(np.mean(values_array)), "ci_lower": float(lower), "ci_upper": float(upper)}


def parse_seed_arg(seeds_arg: str) -> List[str]:
    return [seed.strip() for seed in seeds_arg.split(",") if seed.strip()]


def detect_seed_dirs(results_dir: Path, seeds_arg: str) -> Dict[str, Path]:
    seed_dirs = sorted(results_dir.glob("seed_*"))
    # Check if there are also direct fold dirs (seed not specified, e.g. original v3)
    has_direct_folds = any(results_dir.glob("fold_*"))

    if not seed_dirs and not has_direct_folds:
        return {"default": results_dir}

    available: Dict[str, Path] = {}
    # Direct fold dirs treated as "default" seed
    if has_direct_folds:
        available["default"] = results_dir
    for seed_dir in seed_dirs:
        available[seed_dir.name] = seed_dir

    desired_seeds = parse_seed_arg(seeds_arg)
    desired_keys: List[str] = []
    for seed in desired_seeds:
        if f"seed_{seed}" in available:
            desired_keys.append(f"seed_{seed}")
        elif seed == "42" and "default" in available:
            # seed_42 requested but results are in direct fold dirs (no seed subdir)
            desired_keys.append("default")
    # Also include "default" if present and no specific seeds matched it
    if "default" in available and "default" not in desired_keys:
        desired_keys.insert(0, "default")

    if not desired_keys:
        return available

    missing = [f"seed_{s}" for s in desired_seeds if f"seed_{s}" not in available and not (s == "42" and "default" in available)]
    for seed in missing:
        print(f"Warning: requested {seed} not found in {results_dir}")

    return {k: available[k] for k in desired_keys if k in available}


def infer_ablation_name(cfg: Any, results_dir: Path) -> str:
    checkpoint_dir = getattr(cfg, "logging", {}).get("checkpoint_dir") if hasattr(cfg, "logging") else None
    if checkpoint_dir:
        return Path(checkpoint_dir).parent.name
    if results_dir.name == "checkpoints":
        return results_dir.parent.name
    return results_dir.name


def format_value(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.3f}"


def summarize_metrics(
    per_seed_per_fold: Dict[str, Dict[str, Dict[str, float]]],
    metrics: List[str],
    n_folds: int,
    n_bootstrap: int,
    ci: float,
) -> Dict[str, Dict[str, float]]:
    aggregated = {}
    for metric in metrics:
        fold_means = []
        for fold_id in range(n_folds):
            fold_key = f"fold_{fold_id}"
            values = []
            for seed_results in per_seed_per_fold.values():
                if fold_key not in seed_results:
                    continue
                value = seed_results[fold_key].get(metric)
                if value is None or np.isnan(value):
                    continue
                values.append(value)
            if values:
                fold_means.append(float(np.mean(values)))
        aggregated[metric] = bootstrap_ci(fold_means, n_bootstrap=n_bootstrap, ci=ci)
    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ablation experiments with CI")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    dataset = FabaDroughtDataset(cfg)

    cv = LogoCV(
        plant_metadata_df=dataset.plant_metadata,
        n_folds=cfg.training.cv.n_folds,
        stratify_col=cfg.training.cv.stratify_by,
        seed=cfg.training.cv.seed,
    )

    seed_dirs = detect_seed_dirs(results_dir, args.seeds)
    per_seed_per_fold: Dict[str, Dict[str, Dict[str, float]]] = {}

    for seed_name, seed_dir in seed_dirs.items():
        per_seed_per_fold[seed_name] = {}
        for fold_id, (_, _, test_idx) in enumerate(cv.split()):
            checkpoint_file = seed_dir / f"fold_{fold_id}" / "best_model_state.pt"
            if not checkpoint_file.exists():
                print(f"Warning: missing {checkpoint_file}")
                continue

            print(f"Evaluating {seed_name} fold {fold_id}/{cfg.training.cv.n_folds}...")
            fold_results = evaluate_fold(
                cfg=cfg,
                dataset=dataset,
                fold_id=fold_id,
                test_indices=test_idx,
                checkpoint_root=seed_dir,
            )
            per_seed_per_fold[seed_name][f"fold_{fold_id}"] = fold_results

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "onset_mae",
        "mean_onset_error",
    ]
    aggregated = summarize_metrics(
        per_seed_per_fold=per_seed_per_fold,
        metrics=metrics,
        n_folds=cfg.training.cv.n_folds,
        n_bootstrap=10000,
        ci=0.95,
    )

    ablation_name = infer_ablation_name(cfg, results_dir)
    seed_values = []
    for seed in seed_dirs.keys():
        if seed.startswith("seed_"):
            seed_values.append(seed.replace("seed_", ""))
        else:
            seed_values.append(seed)

    print(f"=== Ablation: {ablation_name} ===")
    print(f"Seeds: {', '.join(seed_values)}\n")
    print("| Metric          | Mean   | 95% CI          |")
    print("|-----------------|--------|-----------------|")

    display_rows = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("AUC", "auc"),
        ("Onset MAE (d)", "onset_mae"),
        ("Mean Error (d)", "mean_onset_error"),
    ]
    for label, key in display_rows:
        stats = aggregated.get(key, {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")})
        mean = format_value(stats["mean"])
        lower = format_value(stats["ci_lower"])
        upper = format_value(stats["ci_upper"])
        print(f"| {label:<15} | {mean:<6} | [{lower}, {upper}]  |")

    results_payload = {
        "ablation_name": ablation_name,
        "seeds": seed_values,
        "per_seed_per_fold": per_seed_per_fold,
        "aggregated": aggregated,
    }

    results_path = output_dir / f"{ablation_name}_ablation_evaluation.json"
    with open(results_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
