"""
Helpers shared by the paper-format rerun scripts (run_exp1.py,
run_exp1_controlled.py, run_exp2.py).

Everything here is pure NumPy; the torch/gpytorch imports stay inside the
scripts that need them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SPLITS = ("train", "valid", "test")
EVAL_SPLITS = ("test", "valid")


def load_split(data_root: str, dataset: str, split: str):
    """Load <data_root>/<dataset>/<split>/{projected_512,y_embeddings}.npy."""
    base = Path(data_root) / dataset / split
    X = np.load(base / "projected_512.npy").astype(np.float32, copy=False)
    y = np.load(base / "y_embeddings.npy").astype(np.float32, copy=False).squeeze()
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"{split}: {X.shape[0]} embeddings vs {y.shape[0]} labels -- misaligned"
        )
    return X, y


def stratified_subsample(rng, X, y, size):
    """Class-balanced subsample, matching the 'stratified' note on the pcam row."""
    if size is None or size >= X.shape[0]:
        return X, y
    classes = np.unique(y)
    per_class = size // len(classes)
    picked = []
    for cls in classes:
        idx = np.flatnonzero(y == cls)
        take = min(per_class, idx.size)
        picked.append(rng.choice(idx, size=take, replace=False))
    keep = np.sort(np.concatenate(picked))
    return X[keep], y[keep]


def experiment_name(base: str, split: str) -> str:
    """Directory / file prefix for an experiment evaluated on a given split.

    The test split keeps the bare name ("exp2") so existing batches still
    aggregate unchanged; any other split is suffixed ("exp2_valid").
    """
    if split not in EVAL_SPLITS:
        raise ValueError(f"unsupported evaluation split: {split}")
    return base if split == "test" else f"{base}_{split}"


def preds_path(run_dir, dataset: str, experiment: str, seed: int) -> Path:
    path = Path(run_dir) / dataset / f"{experiment}_seed{seed}_preds.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def artifact_path(run_dir, dataset: str, name: str) -> Path:
    """Per-seed model / posterior artifacts live next to the preds files."""
    path = Path(run_dir) / dataset / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def point_prediction_fields(metrics: dict, y_true) -> dict:
    """Schema columns for a method that yields a single predictive probability
    per point (SVGP). Posterior-expected quantities collapse onto the point
    estimate because there is no sample spread."""
    elpd_mean = metrics["elpd_mean"]
    nll = metrics["negative_log_likelihood_mean"]
    return {
        "log_likelihood_mean": elpd_mean,
        "negative_log_likelihood_mean": nll,
        "negative_log_loss": nll,
        "mean_predictive_log_likelihood": elpd_mean,
        "predictive_likelihood": float(np.exp(elpd_mean)),
        "elpd": metrics["elpd"],
        "elpd_mean": elpd_mean,
        "pell": elpd_mean,
        "pell_mean": elpd_mean,
        "posterior_expected_log_loss": nll,
        "posterior_log_loss_mean": nll,
        "posterior_total_log_loss": -metrics["elpd"],
        **_classification_fields(metrics, y_true),
    }


def sample_prediction_fields(metrics: dict, y_true) -> dict:
    """Schema columns for a method with posterior probability samples (HMC)."""
    return {
        "log_likelihood_mean": metrics["elpd_mean"],
        "negative_log_likelihood_mean": metrics["negative_log_likelihood_mean"],
        "negative_log_loss": metrics["negative_log_likelihood_mean"],
        "mean_predictive_log_likelihood": metrics["elpd_mean"],
        "elpd": metrics["elpd"],
        "elpd_mean": metrics["elpd_mean"],
        "pell": metrics["pell"],
        "pell_mean": metrics["pell_mean"],
        "posterior_expected_log_loss": metrics["posterior_expected_log_loss"],
        **_classification_fields(metrics, y_true),
    }


def _classification_fields(metrics: dict, y_true) -> dict:
    return {
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "accuracy": metrics["accuracy"],
        "brier": metrics["brier"],
        "ece": metrics["ece"],
        "number_errors": metrics["number_errors"],
        "tp": metrics["TP"],
        "fp": metrics["FP"],
        "tn": metrics["TN"],
        "fn": metrics["FN"],
        "sensitivity_tpr": metrics["sensitivity_TPR"],
        "specificity_tnr": metrics["specificity_TNR"],
        "true_positive_rate": metrics["sensitivity_TPR"],
        "true_negative_rate": metrics["specificity_TNR"],
        "false_positive_rate": metrics["FPR"],
        "false_negative_rate": metrics["FNR"],
        "target_positive_rate": float(np.mean(y_true)),
    }
