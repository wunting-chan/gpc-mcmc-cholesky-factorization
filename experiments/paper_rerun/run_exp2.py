"""
Paper-format rerun of Experiment 2: RPCholesky factor + low-rank HMC GP
classification, evaluated on one or more held-out splits.

One invocation == one seed. Emits one JSON row per evaluated split in the
146-column paper schema ("exp2" for test, "exp2_valid" for validation);
scripts/paper_rerun_aggregate.py averages the seeds afterwards.

The RPCholesky factor is rebuilt on every seed on purpose: pivot selection is
randomized, so holding one factor fixed across runs would understate the
method's true run-to-run spread.

After HMC the posterior (nu samples, pivots, bandwidth, K_PP, F_P) is saved to
<run-dir>/<dataset>/exp2_seed<N>_posterior.npz. Pass it back with
--posterior-file to score another split without re-running the factor and the
sampler (~20 min saved per seed). run_exp1_controlled.py reads the same file
for its pivots.

Pure NumPy/SciPy -- no GPU is used or needed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
for path in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import common  # noqa: E402
import schema  # noqa: E402
from my_cholesky.arpcholesky import arpcholesky  # noqa: E402
from my_cholesky.matrix import KernelMatrix  # noqa: E402
from predictive_metrics2 import (  # noqa: E402
    evaluate_binary_probabilistic_predictions,
    summarize_predictive_distribution,
)

# Reused verbatim from the original experiment -- not reimplemented.
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
from exp2 import (  # noqa: E402
    compute_tau_emcee,
    run_hmc,
    sample_predictive_probabilities_pivots,
)

load_split = common.load_split

POSTERIOR_SCALARS = (
    "bandwidth", "actual_rank", "relative_trace_error", "kernel_queries",
    "trace_K", "trace_FFt", "pivot_consistency", "kernel_time", "cholesky_time",
    "factor_time", "factor_MB", "initial_step", "accept_rate", "step_size",
    "warmup_time", "sampling_time", "tau_logp", "tau_nu_mean", "n_train",
    "max_train",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["pcam", "camelyon17"])
    p.add_argument(
        "--data-root",
        required=True,
        help="Directory holding <dataset>/{train,valid,test}/projected_512.npy",
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--run-dir", default="data/paper_rerun/runs")
    p.add_argument(
        "--eval-splits",
        nargs="+",
        default=["test"],
        choices=list(common.EVAL_SPLITS),
        help="Splits to score. 'test' -> exp2/, 'valid' -> exp2_valid/.",
    )
    p.add_argument(
        "--posterior-file",
        default=None,
        help="exp2_seed<N>_posterior.npz from an earlier run of this seed. "
             "Skips the factor and HMC and only runs predictive sampling.",
    )

    # Defaults below reproduce the recorded rows exactly.
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--arpcholesky-b", type=int, default=10)
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--n-warmup", type=int, default=1000)
    p.add_argument("--n-leapfrog", type=int, default=25)
    p.add_argument("--hmc-step-constant", type=float, default=0.05)
    p.add_argument("--target-accept", type=float, default=0.8)
    p.add_argument("--prediction-batch-size", type=int, default=512)
    p.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Subsample the training split (default: use all rows).",
    )
    return p.parse_args()


def build_posterior(args, X_train, y_train) -> dict:
    """Kernel bandwidth, RPCholesky factor and HMC posterior for this seed."""
    n_train = X_train.shape[0]

    t0 = time.perf_counter()
    A = KernelMatrix(X_train, kernel="gaussian", bandwidth="approx_median")
    kernel_time = time.perf_counter() - t0
    bandwidth = float(A.bandwidth)

    t0 = time.perf_counter()
    lra = arpcholesky(A, k=args.k, b=args.arpcholesky_b, seed=args.seed)
    cholesky_time = time.perf_counter() - t0

    F = np.asarray(lra.get_left_factor()).astype(np.float64, copy=False)  # (N, k)
    pivot_indices = np.asarray(lra.get_indices(), dtype=np.int64)
    actual_rank = int(F.shape[1])

    trace_K = float(A.trace())
    trace_FFt = float(np.sum(F * F))
    rel_trace_error = (trace_K - trace_FFt) / trace_K if trace_K > 0 else float("nan")
    kernel_queries = int(A.num_queries())

    K_pivots = np.asarray(A[pivot_indices, pivot_indices], dtype=np.float64)
    K_pivots = 0.5 * (K_pivots + K_pivots.T)  # symmetrize, as the original does
    F_pivots = F[pivot_indices, :]
    pivot_consistency = float(np.max(np.abs(F_pivots @ F_pivots.T - K_pivots)))
    print(f"  factor k={actual_rank} in {cholesky_time:.2f}s  "
          f"rel_trace_err={rel_trace_error:.4e}  pivot_max={pivot_consistency:.2e}")

    initial_step = args.hmc_step_constant / (args.k ** 0.25)
    hmc = run_hmc(
        factor=F,
        y=y_train,
        n_samples=args.n_samples,
        n_warmup=args.n_warmup,
        seed=args.seed,
        initial_step_size=initial_step,
        n_leapfrog=args.n_leapfrog,
        target_accept=args.target_accept,
        adapt_step_size=True,
    )
    nu_samples = hmc["nu_samples"]
    tau_logp = float(compute_tau_emcee(hmc["logp_trace"]))
    tau_nu_mean = float(compute_tau_emcee(np.mean(nu_samples, axis=1)))
    print(f"  HMC accept={hmc['accept_rate']:.3f}  step={hmc['step_size']:.6f}  "
          f"tau_logp={tau_logp:.2f}")

    return {
        "nu_samples": nu_samples,
        "logp_trace": hmc["logp_trace"],
        "pivot_indices": pivot_indices,
        "X_pivots": X_train[pivot_indices].astype(np.float32, copy=False),
        "F_pivots": F_pivots,
        "K_pivots": K_pivots,
        "bandwidth": bandwidth,
        "actual_rank": actual_rank,
        "relative_trace_error": rel_trace_error,
        "kernel_queries": kernel_queries,
        "trace_K": trace_K,
        "trace_FFt": trace_FFt,
        "pivot_consistency": pivot_consistency,
        "kernel_time": kernel_time,
        "cholesky_time": cholesky_time,
        "factor_time": kernel_time + cholesky_time,
        "factor_MB": F.nbytes / 1e6,
        "initial_step": initial_step,
        "accept_rate": hmc["accept_rate"],
        "step_size": hmc["step_size"],
        "warmup_time": hmc["warmup_time"],
        "sampling_time": hmc["sampling_time"],
        "tau_logp": tau_logp,
        "tau_nu_mean": tau_nu_mean,
        "n_train": n_train,
        "max_train": -1 if args.max_train is None else int(args.max_train),
        "source": "fresh",
    }


def save_posterior(post: dict, path: Path) -> None:
    arrays = {k: post[k] for k in ("nu_samples", "logp_trace", "pivot_indices",
                                   "X_pivots", "F_pivots", "K_pivots")}
    scalars = {k: np.asarray(post[k]) for k in POSTERIOR_SCALARS}
    np.savez(path, **arrays, **scalars)


def load_posterior(path: str) -> dict:
    d = np.load(path)
    post = {k: d[k] for k in ("nu_samples", "logp_trace", "pivot_indices",
                              "X_pivots", "F_pivots", "K_pivots")}
    for k in POSTERIOR_SCALARS:
        post[k] = d[k].item()
    post["source"] = str(path)
    return post


def main() -> None:
    args = parse_args()
    wall_t0 = time.perf_counter()

    # KernelMatrix's "approx_median" bandwidth draws from the legacy global RNG,
    # so seed that too or the bandwidth is not reproducible for a given seed.
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    t0 = time.perf_counter()
    eval_sets = {s: load_split(args.data_root, args.dataset, s) for s in args.eval_splits}
    if args.posterior_file:
        post = load_posterior(args.posterior_file)
        X_train = y_train = None
        feature_dim = int(post["X_pivots"].shape[1])
        print(f"[exp2/{args.dataset}/seed{args.seed}] reusing posterior "
              f"{args.posterior_file} (n_train={post['n_train']})")
    else:
        X_train, y_train = load_split(args.data_root, args.dataset, "train")
        if args.max_train is not None and args.max_train < X_train.shape[0]:
            idx = np.sort(rng.choice(X_train.shape[0], size=args.max_train, replace=False))
            X_train, y_train = X_train[idx], y_train[idx]
        feature_dim = int(X_train.shape[1])
        print(f"[exp2/{args.dataset}/seed{args.seed}] train={X_train.shape} "
              + " ".join(f"{s}={eval_sets[s][0].shape}" for s in args.eval_splits))
    data_loading_time = time.perf_counter() - t0

    if not args.posterior_file:
        post = build_posterior(args, X_train, y_train)
        del X_train  # the (N, k) factor is gone; only the pivot rows are kept
        post_path = common.artifact_path(
            args.run_dir, args.dataset, f"exp2_seed{args.seed}_posterior.npz"
        )
        save_posterior(post, post_path)
        print(f"  saved posterior to {post_path}")

    n_train = int(post["n_train"])
    max_train = None if int(post["max_train"]) < 0 else int(post["max_train"])
    k_actual = int(post["pivot_indices"].shape[0])
    total_mcmc = post["sampling_time"]
    ess_logp = args.n_samples / post["tau_logp"] if post["tau_logp"] > 0 else float("nan")

    # --- assemble the shared part of the 146-column row ------------------
    base_row = schema.blank_row()
    base_row.update(schema.slurm_context())
    base_row.update({
        "experiment": "exp2",
        "dataset": args.dataset,
        "method_name": "RPChol+HMC",
        "sampler": "HMC",
        "script_path": "experiments/paper_rerun/run_exp2.py",
        "run_timestamp_utc": schema.utc_now(),
        "code_ref": schema.git_commit(),
        "dataset_sources": f"['{args.dataset}']",
        "embedding_source": f"{args.dataset}-projected512",
        "embedding_variant": f"paper_rerun_k={args.k}",
        "feature_dim": feature_dim,
        "n_train": n_train,
        "max_train_samples": "full" if max_train is None else max_train,
        "max_test_samples": "full",
        "seed": args.seed,
        "standardize": 0,
        "kernel": "rbf",
        "kernel_bandwidth": post["bandwidth"],
        "k": args.k,
        "arpcholesky_b": args.arpcholesky_b,
        "actual_rank": post["actual_rank"],
        "relative_trace_error": post["relative_trace_error"],
        "kernel_entries_queried": post["kernel_queries"],
        "factor_MB": post["factor_MB"],
        "trace_K": post["trace_K"],
        "trace_FF_T": post["trace_FFt"],
        "pivot_consistency_max": post["pivot_consistency"],
        "n_samples": args.n_samples,
        "n_warmup": args.n_warmup,
        "n_leapfrog": args.n_leapfrog,
        "step_size": post["initial_step"],
        "initial_step": post["initial_step"],
        "final_step_size": post["step_size"],
        "target_accept": args.target_accept,
        "adapt": "True",
        "accept_rate": post["accept_rate"],
        "tau_logp": post["tau_logp"],
        "tau_nu_mean": post["tau_nu_mean"],
        "ess_logp": ess_logp,
        "ess_per_sec": ess_logp / total_mcmc if total_mcmc > 0 else float("nan"),
        # timing
        "data_loading_time_sec": data_loading_time,
        "kernel_time_sec": post["kernel_time"],
        "cholesky_time_sec": post["cholesky_time"],
        "factor_time_sec": post["factor_time"],
        "warmup_time_sec": post["warmup_time"],
        "sampling_time_sec": post["sampling_time"],
        "total_mcmc_time_sec": total_mcmc,
        # HMC's analogue of "training" is warmup + sampling. Recorded under the
        # same field names SVGP uses so the two are comparable in the CSV.
        "fit_or_train_time_sec": post["warmup_time"] + post["sampling_time"],
        "train_time_sec": post["warmup_time"] + post["sampling_time"],
        "timing_scope": (
            "factor, warmup, sampling, predictive_sampling "
            "(post-warmup MCMC time = sampling_time_sec)"
        ),
    })
    if "valid" in eval_sets:
        base_row["n_val"] = int(eval_sets["valid"][0].shape[0])

    # --- predictive distribution, one split at a time ---------------------
    # sample_predictive_probabilities_pivots only ever touches the pivot rows of
    # F and X_train, so hand it those rows with identity pivot indices. This is
    # numerically identical to passing the full arrays and lets a saved
    # posterior be scored without the (N, k) factor.
    identity_pivots = np.arange(k_actual, dtype=np.int64)
    for split in args.eval_splits:
        X_ev, y_ev = eval_sets[split]
        t0 = time.perf_counter()
        pred = sample_predictive_probabilities_pivots(
            F=post["F_pivots"],
            X_train=post["X_pivots"],
            X_test=X_ev,
            pivot_indices=identity_pivots,
            K_pp=post["K_pivots"],
            nu_samples=post["nu_samples"],
            bandwidth=post["bandwidth"],
            batch_size=args.prediction_batch_size,
            seed=args.seed + 999,
        )
        inference_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        summary = summarize_predictive_distribution(
            pred["p_samples"], pred["latent_samples"]
        )
        metrics = evaluate_binary_probabilistic_predictions(
            y_true=y_ev,
            p_pred=summary["prob_mean"],
            p_samples=pred["p_samples"],
            latent_samples=pred["latent_samples"],
        )
        evaluation_time = time.perf_counter() - t0
        total_wall = time.perf_counter() - wall_t0

        exp_name = common.experiment_name("exp2", split)
        row = dict(base_row)
        row.update({
            "record_id": f"rerun-{args.dataset}-{exp_name}-k{args.k}-lf{args.n_leapfrog}-seed{args.seed}",
            "notes": (
                f"Paper-format rerun, seed={args.seed}, evaluated on the {split} "
                f"split. Posterior: {post['source']} (pivots are randomized per "
                f"seed). Pivot |F_P F_P^T - K_PP| max={post['pivot_consistency']:.2e}. "
                f"Embeddings: projected_512 (PCA of the 768-d source), so "
                f"feature_dim={feature_dim} not 768."
            ),
            "n_test": int(X_ev.shape[0]),
            "prob_min": float(np.min(summary["prob_mean"])),
            "prob_mean": float(np.mean(summary["prob_mean"])),
            "prob_max": float(np.max(summary["prob_mean"])),
            "latent_mean_min": float(np.min(summary["latent_mean"])),
            "latent_mean_mean": float(np.mean(summary["latent_mean"])),
            "latent_mean_max": float(np.max(summary["latent_mean"])),
            "latent_var_min": float(np.min(summary["latent_variance"])),
            "latent_var_mean": float(np.mean(summary["latent_variance"])),
            "latent_var_max": float(np.max(summary["latent_variance"])),
            "predictive_sampling_time_sec": inference_time,
            "inference_time_sec": inference_time,
            "prediction_time_sec": inference_time,
            "evaluation_time_sec": evaluation_time,
            "total_pipeline_time_sec": total_wall,
            "total_runtime_sec": total_wall,
        })
        # Metric helper uses its own key spellings; map onto the CSV schema.
        row.update(common.sample_prediction_fields(metrics, y_ev))

        path = schema.write_run(row, args.run_dir, args.dataset, exp_name, args.seed)

        # Per-point arrays for reliability diagrams. The full p_samples matrix
        # is n_samples x n_test and far too large to keep, so store the
        # posterior summaries alongside the mean.
        pred_path = common.preds_path(args.run_dir, args.dataset, exp_name, args.seed)
        np.savez(
            pred_path,
            predictive_prob=summary["prob_mean"],
            y_test=y_ev,
            prob_std=summary["prob_std"],
            latent_mean=summary["latent_mean"],
            latent_std=summary["latent_std"],
        )
        del pred, summary
        print(f"  [{split}] saved predictions to {pred_path}")
        print(f"  [{split}] accuracy={metrics['accuracy']:.6f}  "
              f"auroc={metrics['auroc']:.6f}  ece={metrics['ece']:.6f}  "
              f"total={total_wall:.2f}s")
        print(f"  [{split}] wrote {path}")


if __name__ == "__main__":
    main()
