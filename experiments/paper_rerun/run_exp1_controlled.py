"""
Controlled SVGP: GPyTorch SVGP with the SAME kernel and the SAME inducing
points as the RPCholesky+HMC run (run_exp2.py) for the same seed, so the only
difference between the two methods is variational inference vs HMC.

What is held fixed (not optimized):
  * inducing locations   = X_train[pivots] where pivots are the k RPCholesky
                           pivot indices for this seed (M = k = 200)
  * RBF lengthscale      = the HMC run's kernel bandwidth (the library's
                           Gaussian kernel exp(-0.5 d^2 / bw^2) is exactly
                           GPyTorch's RBF with lengthscale = bw)
  * RBF outputscale      = 1.0
  * prior mean           = 0 (HMC's prior is f = F nu, nu ~ N(0, I))
What is optimized: only the variational distribution q(u), i.e.
  variational_mean and chol_variational_covar, with Adam.

Pivots and bandwidth come from --pivots-file (written by run_exp2.py as
exp2_seed<N>_posterior.npz) when given; otherwise they are rebuilt here with
the exact RNG sequence run_exp2.py uses, which reproduces that seed's pivots
and bandwidth bit-for-bit on the same machine.

One invocation == one seed. Emits one JSON row per evaluated split in the
146-column paper schema under experiment name "exp1c" (test) and "exp1c_valid"
(validation). scripts/paper_rerun_aggregate.py averages the seeds afterwards.

Uses CUDA when available; pass --no-cuda to force CPU (much slower).
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
)

EXPERIMENT = "exp1c"


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
        default=["test", "valid"],
        choices=list(common.EVAL_SPLITS),
        help="Splits to evaluate on. Each gets its own row and preds file.",
    )

    # Where the inducing points / kernel come from.
    p.add_argument(
        "--pivots-file",
        default=None,
        help="exp2_seed<N>_posterior.npz from run_exp2.py (pivot_indices + "
             "bandwidth). Default: rebuild the RPCholesky factor with the "
             "same seed, which gives the same pivots.",
    )
    p.add_argument("--k", type=int, default=200, help="RPCholesky rank = M.")
    p.add_argument("--arpcholesky-b", type=int, default=10)
    p.add_argument(
        "--lengthscale",
        type=float,
        default=None,
        help="Fixed RBF lengthscale. Default: the HMC run's bandwidth for "
             "this seed (~49.6 pcam, ~54.5 camelyon17).",
    )
    p.add_argument("--outputscale", type=float, default=1.0)
    p.add_argument(
        "--likelihood",
        choices=["probit", "logistic"],
        default="probit",
        help="probit = gpytorch BernoulliLikelihood (what the existing SVGP "
             "runs use). logistic = sigmoid link, matching the HMC target "
             "exactly; integrated with Gauss-Hermite quadrature.",
    )

    # Same optimizer settings as run_exp1.py.
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--max-train", type=int, default=None,
                   help="Subsample the training split (smoke tests only).")
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--predict-batch-size", type=int, default=8192)
    return p.parse_args()


def rebuild_pivots(X_train, k, b, seed):
    """Replicate run_exp2.py's factor construction for this seed.

    run_exp2.py calls np.random.seed(seed) once, then KernelMatrix's
    "approx_median" draws 1000 rows from the legacy global RNG, then
    arpcholesky uses default_rng(seed) for proposals and the global RNG for
    rejection draws. Doing the same here in the same order reproduces its
    pivots and bandwidth exactly.
    """
    np.random.seed(seed)
    t0 = time.perf_counter()
    A = KernelMatrix(X_train, kernel="gaussian", bandwidth="approx_median")
    lra = arpcholesky(A, k=k, b=b, seed=seed)
    factor_time = time.perf_counter() - t0
    F = np.asarray(lra.get_left_factor(), dtype=np.float64)
    pivots = np.asarray(lra.get_indices(), dtype=np.int64)
    trace_K = float(A.trace())
    trace_FFt = float(np.sum(F * F))
    info = {
        "bandwidth": float(A.bandwidth),
        "pivot_indices": pivots,
        "actual_rank": int(F.shape[1]),
        "relative_trace_error": (trace_K - trace_FFt) / trace_K if trace_K > 0 else float("nan"),
        "factor_time": factor_time,
        "source": "rebuilt",
    }
    del F, lra, A
    return info


def load_pivots(path):
    d = np.load(path)
    return {
        "bandwidth": float(d["bandwidth"]),
        "pivot_indices": np.asarray(d["pivot_indices"], dtype=np.int64),
        "actual_rank": int(d["pivot_indices"].shape[0]),
        "relative_trace_error": float(d["relative_trace_error"]) if "relative_trace_error" in d else float("nan"),
        "factor_time": float(d["factor_time"]) if "factor_time" in d else float("nan"),
        "source": str(path),
    }


def main() -> None:
    args = parse_args()
    wall_t0 = time.perf_counter()

    import torch
    import gpytorch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)

    t0 = time.perf_counter()
    X_tr, y_tr = common.load_split(args.data_root, args.dataset, "train")
    if args.max_train is not None and args.max_train < X_tr.shape[0]:
        # Same subsampling rule as run_exp2.py --max-train so pivots still line up.
        idx = np.sort(rng.choice(X_tr.shape[0], size=args.max_train, replace=False))
        X_tr, y_tr = X_tr[idx], y_tr[idx]
    eval_sets = {s: common.load_split(args.data_root, args.dataset, s)
                 for s in args.eval_splits}
    data_loading_time = time.perf_counter() - t0
    n_train, feature_dim = X_tr.shape

    # --- inducing points + kernel from the RPCholesky run ----------------
    if args.pivots_file:
        piv = load_pivots(args.pivots_file)
    else:
        piv = rebuild_pivots(X_tr, args.k, args.arpcholesky_b, args.seed)
    pivot_indices = piv["pivot_indices"]
    if pivot_indices.max() >= n_train:
        raise ValueError("pivot index exceeds n_train -- wrong --max-train or pivots file?")
    lengthscale = args.lengthscale if args.lengthscale is not None else piv["bandwidth"]
    num_inducing = int(pivot_indices.shape[0])
    print(f"[{EXPERIMENT}/{args.dataset}/seed{args.seed}] pivots={num_inducing} "
          f"({piv['source']})  bandwidth={piv['bandwidth']:.4f}  "
          f"lengthscale={lengthscale:.4f}  likelihood={args.likelihood}")

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu"
    )
    print(f"  device={device} train={X_tr.shape} "
          + " ".join(f"{s}={eval_sets[s][0].shape}" for s in args.eval_splits))

    X = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Z = X[torch.as_tensor(pivot_indices, device=device)].clone()

    class ControlledSVGP(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points):
            variational_distribution = (
                gpytorch.variational.CholeskyVariationalDistribution(
                    inducing_points.size(0)
                )
            )
            variational_strategy = gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,  # frozen at the pivots
            )
            super().__init__(variational_strategy)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    class LogisticBernoulliLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
        """p(y=1 | f) = sigmoid(f), the link the HMC target uses. Expected log
        prob and the marginal are both Gauss-Hermite integrals over q(f)."""

        def forward(self, function_samples, **kwargs):
            return torch.distributions.Bernoulli(logits=function_samples)

        def marginal(self, function_dist, **kwargs):
            probs = self.quadrature(torch.sigmoid, function_dist)
            return torch.distributions.Bernoulli(probs=probs)

    model = ControlledSVGP(Z).to(device)
    model.covar_module.base_kernel.lengthscale = lengthscale
    model.covar_module.base_kernel.raw_lengthscale.requires_grad_(False)
    model.covar_module.outputscale = args.outputscale
    model.covar_module.raw_outputscale.requires_grad_(False)
    if args.likelihood == "probit":
        likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(device)
    else:
        likelihood = LogisticBernoulliLikelihood().to(device)

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    trainable += [f"likelihood.{n}" for n, p in likelihood.named_parameters() if p.requires_grad]
    expected = {"variational_strategy._variational_distribution.variational_mean",
                "variational_strategy._variational_distribution.chol_variational_covar"}
    if set(trainable) != expected:
        raise RuntimeError(f"unexpected trainable parameters: {trainable}")
    print(f"  trainable: {trainable}")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X.size(0))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model.train()
    likelihood.train()
    t0 = time.perf_counter()
    train_loss = float("nan")
    for epoch in range(args.num_epochs):
        running = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = -mll(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * xb.size(0)
        train_loss = running / X.size(0)
        if (epoch + 1) % max(1, args.num_epochs // 10) == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:>3}/{args.num_epochs}: loss={train_loss:.6f}")
    train_time = time.perf_counter() - t0

    # Sanity: the frozen quantities really did not move.
    assert torch.allclose(model.variational_strategy.inducing_points, Z)
    assert abs(float(model.covar_module.base_kernel.lengthscale) - lengthscale) < 1e-4
    assert abs(float(model.covar_module.outputscale) - args.outputscale) < 1e-6

    model.eval()
    likelihood.eval()

    def predict(X_np):
        probs = []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for start in range(0, len(X_np), args.predict_batch_size):
                xb = torch.tensor(
                    X_np[start : start + args.predict_batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                probs.append(likelihood(model(xb)).probs.detach().cpu().numpy())
        return np.concatenate(probs, axis=0)

    model_path = common.artifact_path(
        args.run_dir, args.dataset, f"{EXPERIMENT}_seed{args.seed}_model.pt"
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "likelihood_state_dict": likelihood.state_dict(),
            "likelihood": args.likelihood,
            "pivot_indices": pivot_indices,
            "lengthscale": lengthscale,
            "outputscale": args.outputscale,
            "seed": args.seed,
            "dataset": args.dataset,
        },
        model_path,
    )
    print(f"  saved model to {model_path}")

    base_row = schema.blank_row()
    base_row.update(schema.slurm_context())
    base_row.update({
        "experiment": EXPERIMENT,
        "dataset": args.dataset,
        "method_name": "SVGP-controlled",
        "sampler": schema.NOT_APPLICABLE,
        "model_architecture": f"gpytorch_svgp_fixed_pivots_{args.likelihood}",
        "script_path": "experiments/paper_rerun/run_exp1_controlled.py",
        "run_timestamp_utc": schema.utc_now(),
        "code_ref": schema.git_commit(),
        "dataset_sources": f"['{args.dataset}']",
        "embedding_source": f"{args.dataset}-projected512",
        "embedding_variant": f"controlled_k={num_inducing}_{args.likelihood}",
        "feature_dim": feature_dim,
        "n_train": n_train,
        "max_train_samples": "full" if args.max_train is None else args.max_train,
        "max_valid_samples": "full",
        "max_test_samples": "full",
        "seed": args.seed,
        "standardize": 0,
        "kernel": "rbf",
        "kernel_bandwidth": lengthscale,
        "lengthscale_initial": lengthscale,
        "outputscale_initial": args.outputscale,
        "k": num_inducing,
        "arpcholesky_b": args.arpcholesky_b,
        "actual_rank": piv["actual_rank"],
        "relative_trace_error": piv["relative_trace_error"],
        "num_inducing": num_inducing,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs_requested": args.num_epochs,
        "epochs_ran": args.num_epochs,
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "train_loss_final": train_loss,
        "data_loading_time_sec": data_loading_time,
        "factor_time_sec": piv["factor_time"],
        "fit_or_train_time_sec": train_time,
        "train_time_sec": train_time,
        "timing_scope": "training, prediction (data loading and pivot "
                        "construction measured separately)",
    })

    for split in args.eval_splits:
        X_ev, y_ev = eval_sets[split]
        t0 = time.perf_counter()
        p_ev = predict(X_ev)
        inference_time = time.perf_counter() - t0
        metrics = evaluate_binary_probabilistic_predictions(y_true=y_ev, p_pred=p_ev)
        total_wall = time.perf_counter() - wall_t0

        exp_name = common.experiment_name(EXPERIMENT, split)
        row = dict(base_row)
        row.update({
            "record_id": f"rerun-{args.dataset}-{exp_name}-k{num_inducing}-seed{args.seed}",
            "notes": (
                f"Controlled SVGP, seed={args.seed}, evaluated on the {split} "
                f"split. Inducing points frozen at the {num_inducing} RPCholesky "
                f"pivots ({piv['source']}); lengthscale fixed at {lengthscale:.4f}, "
                f"outputscale fixed at {args.outputscale}, zero prior mean, "
                f"{args.likelihood} likelihood; only q(u) optimized. pell "
                f"collapses onto elpd (point predictive probabilities)."
            ),
            "n_test": int(X_ev.shape[0]),
            "inference_time_sec": inference_time,
            "prediction_time_sec": inference_time,
            "total_pipeline_time_sec": total_wall,
            "total_runtime_sec": total_wall,
        })
        if "valid" in eval_sets:
            row["n_val"] = int(eval_sets["valid"][0].shape[0])
        row.update(common.point_prediction_fields(metrics, y_ev))

        path = schema.write_run(row, args.run_dir, args.dataset, exp_name, args.seed)
        pred_path = common.preds_path(args.run_dir, args.dataset, exp_name, args.seed)
        np.savez(pred_path, predictive_prob=p_ev, y_test=y_ev)
        print(f"  [{split}] accuracy={metrics['accuracy']:.6f}  auroc={metrics['auroc']:.6f}  "
              f"ece={metrics['ece']:.6f}  nll={metrics['negative_log_likelihood_mean']:.6f}  "
              f"fnr={metrics['FNR']:.6f}")
        print(f"  [{split}] wrote {path} and {pred_path}")

    print(f"  total={time.perf_counter() - wall_t0:.2f}s")


if __name__ == "__main__":
    main()
