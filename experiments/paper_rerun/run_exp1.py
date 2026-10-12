"""
Paper-format rerun of Experiment 1: GPyTorch SVGP binary classification on
precomputed embeddings.

One invocation == one seed. Emits a single JSON row in the 146-column paper
schema; scripts/paper_rerun_aggregate.py averages the seeds afterwards.

Model setup mirrors experiments/exp1_gpytorch_svgp_gpc_embeddings.py
(CholeskyVariationalDistribution + VariationalStrategy with learned inducing
locations, ConstantMean, ScaleKernel(RBFKernel), BernoulliLikelihood,
VariationalELBO, Adam). That script is imported nowhere and left untouched.

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
from predictive_metrics2 import (  # noqa: E402
    evaluate_binary_probabilistic_predictions,
)

load_split = common.load_split
stratified_subsample = common.stratified_subsample

# Both datasets now train on their full splits, so SVGP sees the same data as
# the RPCholesky+HMC run it is compared against. The recorded pcam row used a
# stratified 3000/500 subsample -- reproduce it with --max-train 3000
# --max-valid 500 if you need that number back.
DATASET_DEFAULTS = {
    "pcam": {"max_train": None, "max_valid": None},
    "camelyon17": {"max_train": None, "max_valid": None},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
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
        help="Splits to write metric rows for. 'test' -> exp1/, "
             "'valid' -> exp1_valid/ (in-distribution check for camelyon17).",
    )

    # Defaults below reproduce the recorded rows exactly.
    p.add_argument("--num-inducing", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--lengthscale-init", type=float, default=64.0)
    p.add_argument("--outputscale-init", type=float, default=1.0)
    p.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Override the dataset default training-subsample size.",
    )
    p.add_argument("--max-valid", type=int, default=None)
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--predict-batch-size", type=int, default=8192)
    return p.parse_args()


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

    defaults = DATASET_DEFAULTS[args.dataset]
    max_train = args.max_train if args.max_train is not None else defaults["max_train"]
    max_valid = args.max_valid if args.max_valid is not None else defaults["max_valid"]

    t0 = time.perf_counter()
    X_tr, y_tr = load_split(args.data_root, args.dataset, "train")
    X_va, y_va = load_split(args.data_root, args.dataset, "valid")
    X_te, y_te = load_split(args.data_root, args.dataset, "test")
    X_tr, y_tr = stratified_subsample(rng, X_tr, y_tr, max_train)
    X_va, y_va = stratified_subsample(rng, X_va, y_va, max_valid)
    data_loading_time = time.perf_counter() - t0

    n_train, feature_dim = X_tr.shape
    n_val, n_test = X_va.shape[0], X_te.shape[0]
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu"
    )
    print(f"[exp1/{args.dataset}/seed{args.seed}] device={device} "
          f"train={X_tr.shape} valid={X_va.shape} test={X_te.shape}")

    X = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y = torch.tensor(y_tr, dtype=torch.float32, device=device)

    class SVGPBinaryClassifier(gpytorch.models.ApproximateGP):
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
                learn_inducing_locations=True,
            )
            super().__init__(variational_strategy)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    num_inducing = min(args.num_inducing, X.size(0))
    perm = torch.randperm(X.size(0), device=device)
    model = SVGPBinaryClassifier(X[perm[:num_inducing]].clone()).to(device)
    model.covar_module.base_kernel.lengthscale = args.lengthscale_init
    model.covar_module.outputscale = args.outputscale_init
    likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(device)

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": likelihood.parameters()}],
        lr=args.learning_rate,
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

    # The validation split is always scored (it feeds best_val_auroc); the
    # test split only if requested. Prediction time is measured per split.
    predictions = {}
    pred_times = {}
    for split, X_np in (("valid", X_va), ("test", X_te)):
        if split != "valid" and split not in args.eval_splits:
            continue
        t0 = time.perf_counter()
        predictions[split] = predict(X_np)
        pred_times[split] = time.perf_counter() - t0
    inference_time = sum(pred_times.values())

    valid_metrics = evaluate_binary_probabilistic_predictions(
        y_true=y_va, p_pred=predictions["valid"]
    )

    # Save the trained model so other splits can be scored without retraining.
    model_path = common.artifact_path(
        args.run_dir, args.dataset, f"exp1_seed{args.seed}_model.pt"
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "likelihood_state_dict": likelihood.state_dict(),
            "num_inducing": num_inducing,
            "seed": args.seed,
            "dataset": args.dataset,
        },
        model_path,
    )
    print(f"  saved model to {model_path}")

    base_row = schema.blank_row()
    base_row.update(schema.slurm_context())
    base_row.update({
        "experiment": "exp1",
        "dataset": args.dataset,
        "method_name": "SVGP",
        "sampler": schema.NOT_APPLICABLE,
        "model_architecture": "gpytorch_svgp",
        "script_path": "experiments/paper_rerun/run_exp1.py",
        "run_timestamp_utc": schema.utc_now(),
        "code_ref": schema.git_commit(),
        "dataset_sources": f"['{args.dataset}']",
        "embedding_source": f"{args.dataset}-projected512",
        "embedding_variant": f"ls{args.lengthscale_init}",
        "feature_dim": feature_dim,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "max_train_samples": "full" if max_train is None else max_train,
        "max_valid_samples": "full" if max_valid is None else max_valid,
        "max_test_samples": "full",
        "seed": args.seed,
        "standardize": 0,
        "kernel": "rbf",
        "kernel_bandwidth": args.lengthscale_init,
        "lengthscale_initial": args.lengthscale_init,
        "outputscale_initial": args.outputscale_init,
        "num_inducing": num_inducing,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs_requested": args.num_epochs,
        "epochs_ran": args.num_epochs,
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "best_val_auroc": valid_metrics["auroc"],
        "train_loss_final": train_loss,
        "data_loading_time_sec": data_loading_time,
        "fit_or_train_time_sec": train_time,
        "train_time_sec": train_time,
        "inference_time_sec": inference_time,
        "prediction_time_sec": inference_time,
        "timing_scope": "training, prediction (data loading measured separately)",
    })

    labels = {"valid": y_va, "test": y_te}
    for split in args.eval_splits:
        y_ev = labels[split]
        p_ev = predictions[split]
        metrics = (valid_metrics if split == "valid"
                   else evaluate_binary_probabilistic_predictions(y_true=y_ev, p_pred=p_ev))
        total_wall = time.perf_counter() - wall_t0

        exp_name = common.experiment_name("exp1", split)
        row = dict(base_row)
        row.update({
            "record_id": f"rerun-{args.dataset}-{exp_name}-svgp-seed{args.seed}",
            "notes": (
                f"Paper-format rerun, seed={args.seed}, evaluated on the {split} "
                f"split. pell left unset: SVGP gives point posterior predictive "
                f"probabilities, not probability samples. Embeddings: "
                f"projected_512 (PCA of the 768-d source), so "
                f"feature_dim={feature_dim} not 768."
            ),
            "total_pipeline_time_sec": total_wall,
            "total_runtime_sec": total_wall,
        })
        # SVGP has no posterior probability samples, so the posterior-expected
        # quantities collapse onto the point-estimate log loss.
        row.update(common.point_prediction_fields(metrics, y_ev))

        path = schema.write_run(row, args.run_dir, args.dataset, exp_name, args.seed)

        # Per-point arrays for reliability diagrams. SVGP gives a point
        # predictive probability, so there is no sample spread to store.
        pred_path = common.preds_path(args.run_dir, args.dataset, exp_name, args.seed)
        np.savez(pred_path, predictive_prob=p_ev, y_test=y_ev)
        print(f"  [{split}] saved predictions to {pred_path}")
        print(f"  [{split}] accuracy={metrics['accuracy']:.6f}  "
              f"auroc={metrics['auroc']:.6f}  ece={metrics['ece']:.6f}  "
              f"total={total_wall:.2f}s")
        print(f"  [{split}] wrote {path}")


if __name__ == "__main__":
    main()
