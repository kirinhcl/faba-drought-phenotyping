#!/usr/bin/env python3
# pyright: reportMissingImports=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnusedCallResult=false
"""Train classical ML baselines for per-timestep stress classification."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Callable, Protocol, cast

import numpy as np
import torch
from omegaconf import DictConfig
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.data.dataset import FabaDroughtDataset, ROUND_TO_DAG
from src.training.cv import LogoCV
from src.utils.config import load_config


class Classifier(Protocol):
    def fit(self, X: NDArray[np.float32], y: NDArray[np.int64]) -> object:
        ...

    def predict(self, X: NDArray[np.float32]) -> object:
        ...

    def predict_proba(self, X: NDArray[np.float32]) -> object:
        ...


XGBClassifierType: Callable[..., Classifier] | None = None
has_xgboost = False
try:
    xgb_module = importlib.import_module("xgboost")
    XGBClassifierType = cast(type[Classifier], getattr(xgb_module, "XGBClassifier"))
    has_xgboost = True
except Exception:
    print("Warning: XGBoost not available, skipping")


def bootstrap_ci(values: Iterable[float], n_bootstrap: int = 10000, ci: float = 0.95) -> dict[str, float]:
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


def format_value(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.3f}"


def extract_features(sample: dict[str, object]) -> tuple[list[NDArray[np.float32]], list[int], list[int]]:
    fluorescence = cast(torch.Tensor, sample["fluorescence"])
    environment = cast(torch.Tensor, sample["environment"])
    vi = cast(torch.Tensor, sample["vi"])
    temporal_positions = cast(torch.Tensor, sample["temporal_positions"])
    stress_labels = cast(torch.Tensor, sample["stress_labels"])
    stress_mask = cast(torch.Tensor, sample["stress_mask"])

    features: list[NDArray[np.float32]] = []
    labels: list[int] = []
    valid_indices: list[int] = []

    for t_idx in range(len(stress_mask)):
        if not bool(stress_mask[t_idx]):
            continue
        dag_value = temporal_positions[t_idx].unsqueeze(0)
        feature_vec = torch.cat(
            [fluorescence[t_idx], environment[t_idx], vi[t_idx], dag_value],
            dim=0,
        )
        features.append(feature_vec.numpy())
        labels.append(int(stress_labels[t_idx].item()))
        valid_indices.append(t_idx)

    return features, labels, valid_indices


def build_train_data(
    dataset: FabaDroughtDataset,
    indices: NDArray[np.int64],
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    X: list[NDArray[np.float32]] = []
    y: list[int] = []
    for idx in cast(list[int], indices.tolist()):
        sample = dataset[int(idx)]
        features, labels, _ = extract_features(sample)
        X.extend(features)
        y.extend(labels)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def build_test_data(
    dataset: FabaDroughtDataset,
    indices: NDArray[np.int64],
) -> tuple[NDArray[np.float32], NDArray[np.int64], list[dict[str, object]]]:
    X: list[NDArray[np.float32]] = []
    y: list[int] = []
    plant_records: list[dict[str, object]] = []

    for idx in cast(list[int], indices.tolist()):
        sample = dataset[int(idx)]
        features, labels, valid_indices = extract_features(sample)
        start = len(X)
        X.extend(features)
        y.extend(labels)
        end = len(X)

        plant_records.append({
            "treatment": cast(str, sample["treatment"]),
            "stress_labels": cast(torch.Tensor, sample["stress_labels"]).numpy().astype(int),
            "valid_indices": valid_indices,
            "flat_slice": (start, end),
        })

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), plant_records


def compute_onset_metrics(test_plants: list[dict[str, object]]) -> dict[str, float]:
    results = []

    for plant in test_plants:
        if cast(str, plant["treatment"]) == "WHC-80":
            continue
        stress_labels = cast(NDArray[np.int64], plant["stress_labels"]).astype(int)
        if stress_labels.sum() == 0:
            continue

        stress_values = cast(list[int], stress_labels.tolist())
        stress_indices = [i for i, val in enumerate(stress_values) if val == 1]
        if not stress_indices:
            continue
        true_onset_idx = int(stress_indices[0])
        true_onset_dag = ROUND_TO_DAG[true_onset_idx + 2]

        pred_probs = cast(NDArray[np.float32], plant["pred_probs"]).astype(float)
        pred_values = cast(list[float], pred_probs.tolist())
        pred_stress_indices = [i for i, val in enumerate(pred_values) if val > 0.5]
        if not pred_stress_indices:
            pred_onset_idx = len(pred_probs) - 1
        else:
            pred_onset_idx = int(pred_stress_indices[0])
        pred_onset_dag = ROUND_TO_DAG[pred_onset_idx + 2]

        error = pred_onset_dag - true_onset_dag
        results.append(error)

    if len(results) == 0:
        return {
            "onset_mae": float("nan"),
            "mean_onset_error": float("nan"),
        }

    errors = np.array(results, dtype=float)
    return {
        "onset_mae": float(np.mean(np.abs(errors))),
        "mean_onset_error": float(np.mean(errors)),
    }


def evaluate_fold(
    classifier: Classifier,
    X_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    X_test: NDArray[np.float32],
    y_test: NDArray[np.int64],
    test_plants: list[dict[str, object]],
) -> dict[str, float]:
    classifier.fit(X_train, y_train)

    preds = cast(NDArray[np.int64], classifier.predict(X_test))
    try:
        probs = cast(NDArray[np.float32], classifier.predict_proba(X_test))[:, 1]
    except Exception:
        probs = preds.astype(float)

    accuracy = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        preds,
        average="binary",
        zero_division=0,  # pyright: ignore[reportArgumentType]
    )
    try:
        auc = roc_auc_score(y_test, probs)
    except ValueError:
        auc = float("nan")

    for plant in test_plants:
        start, end = cast(tuple[int, int], plant["flat_slice"])
        stress_labels = cast(NDArray[np.int64], plant["stress_labels"])
        valid_indices = cast(list[int], plant["valid_indices"])
        plant_probs = np.zeros_like(stress_labels, dtype=float)
        probs_list = cast(list[float], probs[start:end].tolist())
        for t_idx, prob in zip(valid_indices, probs_list):
            plant_probs[t_idx] = prob
        plant["pred_probs"] = plant_probs

    onset_metrics = compute_onset_metrics(test_plants)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        **onset_metrics,
    }


def summarize_metrics(
    per_fold: dict[str, dict[str, float]],
    metrics: list[str],
    n_bootstrap: int,
    ci: float,
) -> dict[str, dict[str, float]]:
    aggregated = {}
    for metric in metrics:
        values = [fold_metrics.get(metric, float("nan")) for fold_metrics in per_fold.values()]
        aggregated[metric] = bootstrap_ci(values, n_bootstrap=n_bootstrap, ci=ci)
    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description="Train classical ML baselines")
    _ = parser.add_argument("--config", type=str, default="configs/stress_v3.yaml")
    _ = parser.add_argument("--seed", type=int, default=42)
    _ = parser.add_argument("--output_dir", type=str, default="results/baselines/")
    _ = parser.add_argument("--n_bootstrap", type=int, default=10000)
    class Args(argparse.Namespace):
        config: str = "configs/stress_v3.yaml"
        seed: int = 42
        output_dir: str = "results/baselines/"
        n_bootstrap: int = 10000

    args = cast(Args, parser.parse_args())

    cfg = cast(DictConfig, load_config(args.config))
    dataset = FabaDroughtDataset(cfg)

    cv = LogoCV(
        plant_metadata_df=dataset.plant_metadata,
        n_folds=int(cfg.training.cv.n_folds),
        stratify_col=str(cfg.training.cv.stratify_by),
        seed=int(cfg.training.cv.seed),
    )

    classifiers: dict[str, Classifier] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=args.seed,
        ),
    }

    if has_xgboost and XGBClassifierType is not None:
        classifiers["XGBoost"] = XGBClassifierType(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=1.5,
            random_state=args.seed,
            n_jobs=-1,
            eval_metric="logloss",
        )

    results: dict[str, object] = {
        "seed": args.seed,
        "classifiers": {},
    }

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "onset_mae",
        "mean_onset_error",
    ]

    for name, classifier in classifiers.items():
        print(f"=== {name} (seed={args.seed}) ===")
        per_fold: dict[str, dict[str, float]] = {}
        for fold_id, (train_idx, _, test_idx) in enumerate(cv.split()):
            print(f"Fold {fold_id}/{cfg.training.cv.n_folds}...", end=" ", flush=True)
            X_train, y_train = build_train_data(dataset, train_idx)
            X_test, y_test, test_plants = build_test_data(dataset, test_idx)
            fold_results = evaluate_fold(
                classifier=classifier,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                test_plants=test_plants,
            )
            per_fold[f"fold_{fold_id}"] = fold_results
        print("\n")

        aggregated = summarize_metrics(
            per_fold=per_fold,
            metrics=metrics,
            n_bootstrap=args.n_bootstrap,
            ci=0.95,
        )

        print(f"=== Baseline: {name} (seed={args.seed}) ===\n")
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
        print("")

        cast(dict[str, object], results["classifiers"])[name] = {
            "per_fold": per_fold,
            "aggregated": aggregated,
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"baselines_seed_{args.seed}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
