"""
CAMELYON17 HMC leapfrog sweep — local CPU script.

Reproduces and extends the Appendix A Table 4 sweep for CAMELYON17.
Saves nu_samples and logp_trace per config so you can inspect mixing
quality beyond the scalar tau summary.

Usage:
    python sweep_leapfrog_cam17.py \
        --data-root /path/to/data \
        --out-dir sweep_results

Assumes the repo is cloned and on PYTHONPATH (or run from repo root).
Pure NumPy — no GPU needed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---- path setup: adjust if running from outside the repo ----
SCRIPT_DIR = Path(__file__).resolve().parent
# Try to find the repo root from common locations
for candidate in [
    SCRIPT_DIR,
    SCRIPT_DIR.parent,
    Path.cwd(),
]:
    if (candidate / "experiments" / "exp2.py").exists():
        REPO_ROOT = candidate
        break
    if (candidate / "exp2.py").exists():
        REPO_ROOT = candidate.parent
        break
else:
    REPO_ROOT = Path.cwd()

for p in [
    str(REPO_ROOT),
    str(REPO_ROOT / "src"),
    str(REPO_ROOT / "experiments"),
    str(REPO_ROOT / "experiments" / "paper_rerun"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from my_cholesky.arpcholesky import arpcholesky
from my_cholesky.matrix import KernelMatrix
from exp2 import run_hmc, compute_tau_emcee

# Try to import common from paper_rerun; fall back to manual loading
try:
    import common as rerun_common
    load_split = rerun_common.load_split
except ImportError:
    def load_split(data_root, dataset, split):
        base = Path(data_root) / dataset / split
        X = np.load(base / "projected_512.npy").astype(np.float32, copy=False)
        y = np.load(base / "y_embeddings.npy").astype(np.float32, copy=False).squeeze()
        return X, y


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True,
                   help="Directory with camelyon17/{train,valid,test}/projected_512.npy")
    p.add_argument("--dataset", default="camelyon17",
                   choices=["pcam", "camelyon17"])
    p.add_argument("--out-dir", default="sweep_results",
                   help="Where to save .npz outputs and the summary JSON")
    p.add_argument("--max-train", type=int, default=50_000,
                   help="Subsample training set (default: 50000)")
    p.add_argument("--seed", type=int, default=0)

    # Fixed HMC parameters
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--arpcholesky-b", type=int, default=10)
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--n-warmup", type=int, default=1000)
    p.add_argument("--step-constant", type=float, default=0.05)
    p.add_argument("--target-accept", type=float, default=0.8)

    # Sweep values
    p.add_argument(
        "--leapfrog-values",
        type=int,
        nargs="+",
        default=[10, 16, 25, 35, 50],
        help="n_leapfrog values to sweep (default: 10 16 25 35 50)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Seed both RNGs for reproducibility (KernelMatrix's approx_median
    # uses the legacy global RNG).
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # ---- load and subsample ----
    print(f"Loading {args.dataset} train split...")
    X_train, y_train = load_split(args.data_root, args.dataset, "train")
    print(f"  full train: {X_train.shape}")

    if args.max_train is not None and args.max_train < X_train.shape[0]:
        idx = np.sort(rng.choice(X_train.shape[0], size=args.max_train, replace=False))
        X_train, y_train = X_train[idx], y_train[idx]
        print(f"  subsampled to: {X_train.shape}")

    n_train = X_train.shape[0]

    # ---- build kernel + RPCholesky factor (shared across sweep) ----
    print("Building kernel matrix...")
    t0 = time.perf_counter()
    A = KernelMatrix(X_train, kernel="gaussian", bandwidth="approx_median")
    kernel_time = time.perf_counter() - t0
    bandwidth = float(A.bandwidth)
    print(f"  bandwidth (approx_median) = {bandwidth:.4f}  ({kernel_time:.2f}s)")

    print(f"Running RPCholesky (k={args.k}, b={args.arpcholesky_b})...")
    t0 = time.perf_counter()
    lra = arpcholesky(A, k=args.k, b=args.arpcholesky_b, seed=args.seed)
    cholesky_time = time.perf_counter() - t0

    F = np.asarray(lra.get_left_factor()).astype(np.float64, copy=False)
    pivot_indices = np.asarray(lra.get_indices(), dtype=np.int64)
    actual_rank = F.shape[1]

    trace_K = float(A.trace())
    trace_FFt = float(np.sum(F * F))
    rel_trace_err = (trace_K - trace_FFt) / trace_K if trace_K > 0 else float("nan")
    print(f"  rank={actual_rank}  rel_trace_err={rel_trace_err:.4e}  ({cholesky_time:.2f}s)")

    initial_step = args.step_constant / (args.k ** 0.25)
    print(f"  initial_step = {initial_step:.6f}")
    print()

    # ---- sweep ----
    results = []
    header = f"{'n_leapfrog':>10}  {'tau_nu':>10}  {'tau_logp':>10}  " \
             f"{'accept':>10}  {'step_size':>12}  {'wall_time':>12}"
    print(header)
    print("-" * len(header))

    for n_lf in args.leapfrog_values:
        print(f"\n>>> n_leapfrog = {n_lf}", flush=True)
        t_start = time.perf_counter()

        hmc = run_hmc(
            factor=F,
            y=y_train,
            n_samples=args.n_samples,
            n_warmup=args.n_warmup,
            seed=args.seed,
            initial_step_size=initial_step,
            n_leapfrog=n_lf,
            target_accept=args.target_accept,
            adapt_step_size=True,
            metric="laplace",
        )

        wall_time = time.perf_counter() - t_start
        nu_samples = hmc["nu_samples"]
        logp_trace = hmc["logp_trace"]

        # Autocorrelation diagnostics
        tau_nu = float(compute_tau_emcee(nu_samples))
        tau_logp = float(compute_tau_emcee(logp_trace))

        # Per-component tau for the full nu vector
        try:
            tau_per_component = np.array([
                float(compute_tau_emcee(nu_samples[:, j:j+1]))
                for j in range(min(actual_rank, 20))  # first 20 components
            ])
        except Exception:
            tau_per_component = np.array([])

        accept_rate = hmc["accept_rate"]
        step_size = hmc["step_size"]

        row = {
            "n_leapfrog": n_lf,
            "tau_nu": tau_nu,
            "tau_logp": tau_logp,
            "accept_rate": accept_rate,
            "step_size": step_size,
            "warmup_time": hmc["warmup_time"],
            "sampling_time": hmc["sampling_time"],
            "wall_time": wall_time,
            "ess_nu": args.n_samples / max(tau_nu, 1.0) if np.isfinite(tau_nu) else float("nan"),
            "ess_logp": args.n_samples / max(tau_logp, 1.0) if np.isfinite(tau_logp) else float("nan"),
        }
        results.append(row)

        print(f"{n_lf:>10}  {tau_nu:>10.2f}  {tau_logp:>10.2f}  "
              f"{accept_rate:>10.3f}  {step_size:>12.6f}  {wall_time:>12.1f}")

        # Save full diagnostic data per config
        npz_path = out_dir / f"lf{n_lf}_seed{args.seed}.npz"
        np.savez(
            npz_path,
            nu_samples=nu_samples,
            logp_trace=logp_trace,
            tau_per_component=tau_per_component,
            # Scalars
            n_leapfrog=n_lf,
            tau_nu=tau_nu,
            tau_logp=tau_logp,
            accept_rate=accept_rate,
            step_size=step_size,
            warmup_time=hmc["warmup_time"],
            sampling_time=hmc["sampling_time"],
            wall_time=wall_time,
            # Shared across configs
            bandwidth=bandwidth,
            n_train=n_train,
            k=actual_rank,
            seed=args.seed,
            pivot_indices=pivot_indices,
        )
        print(f"  saved to {npz_path}")

    # ---- summary table ----
    print("\n" + "=" * 80)
    print(f"SWEEP SUMMARY — {args.dataset}, N={n_train}, k={actual_rank}, "
          f"bandwidth={bandwidth:.2f}, seed={args.seed}")
    print(f"n_warmup={args.n_warmup}, n_samples={args.n_samples}, "
          f"step_constant={args.step_constant}, target_accept={args.target_accept}")
    print("=" * 80)
    print(f"{'n_leapfrog':>10}  {'tau_nu':>10}  {'tau_logp':>10}  "
          f"{'accept':>10}  {'step_size':>12}  {'wall(s)':>10}  "
          f"{'ESS_nu':>10}  {'ESS_logp':>10}")
    print("-" * 94)
    for r in results:
        print(f"{r['n_leapfrog']:>10}  {r['tau_nu']:>10.2f}  {r['tau_logp']:>10.2f}  "
              f"{r['accept_rate']:>10.3f}  {r['step_size']:>12.6f}  {r['wall_time']:>10.1f}  "
              f"{r['ess_nu']:>10.1f}  {r['ess_logp']:>10.1f}")

    # Save summary as JSON for easy comparison
    summary_path = out_dir / f"sweep_summary_seed{args.seed}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "config": {
                "dataset": args.dataset,
                "n_train": n_train,
                "k": actual_rank,
                "bandwidth": bandwidth,
                "n_warmup": args.n_warmup,
                "n_samples": args.n_samples,
                "step_constant": args.step_constant,
                "target_accept": args.target_accept,
                "seed": args.seed,
                "rel_trace_err": rel_trace_err,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()