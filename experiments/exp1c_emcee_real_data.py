"""
Experiment 1c: compare three samplers on real-data RPCholesky GP classification factors.

- emcee-based RWM using a Gaussian random-walk proposal
- emcee-based MALA
- a self-contained HMC implementation

This keeps the same non-centered parameterization f = F @ nu used in the
other experiments. emcee drives the RWM and MALA runs; HMC is added directly
because emcee does not provide an HMC kernel.

Example:
    python experiments/exp1c_emcee_real_data.py \
        --factor-dir data/precomputed_factors
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import socket
import sys
import time
from pathlib import Path

import emcee
import numpy as np
from scipy.special import expit
from scipy.stats import norm


# Allow direct script execution without package install.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare emcee and HMC samplers on precomputed low-rank factors."
    )
    parser.add_argument(
        "--factor-dir",
        type=str,
        required=True,
        help="Directory containing factor_k{k}.npy and labels.npy",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[10, 20, 50, 100, 200],
    )
    parser.add_argument("--n-samples", type=int, default=20000)
    parser.add_argument("--n-warmup", type=int, default=200)
    parser.add_argument("--n-seed-repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--dataset-name", type=str, default="real_embeddings")
    parser.add_argument(
        "--likelihood",
        type=str,
        choices=["sigmoid", "probit"],
        default="probit",
        help="Bernoulli likelihood link function.",
    )
    parser.add_argument(
        "--rwm-step-constant",
        type=float,
        default=0.8,
        help="Constant for RWM step size: constant / sqrt(actual_rank).",
    )
    parser.add_argument(
        "--mala-step-constant",
        type=float,
        default=0.243,
        help="Constant for MALA step size: constant / actual_rank^(1/3).",
    )
    parser.add_argument(
        "--hmc-step-constant",
        type=float,
        default=0.05,
        help="Initial HMC step constant: constant / actual_rank^(1/4).",
    )
    parser.add_argument(
        "--hmc-target-accept",
        type=float,
        default=0.8,
        help="Target HMC acceptance probability for dual averaging warmup.",
    )
    parser.add_argument(
        "--no-hmc-adapt-step-size",
        action="store_false",
        dest="hmc_adapt_step_size",
        help="Disable HMC dual averaging step-size adaptation during warmup.",
    )
    return parser.parse_args()


def log_posterior(
    nu: np.ndarray,
    factor: np.ndarray,
    y: np.ndarray,
    likelihood: str,
) -> float:
    """Log posterior for Bernoulli likelihood and standard normal prior."""
    f = factor @ nu
    if likelihood == "sigmoid":
        p = expit(f)
        log_lik = np.sum(y * np.log(p + 1e-10) + (1 - y) * np.log(1 - p + 1e-10))
    elif likelihood == "probit":
        log_lik = np.sum(y * norm.logcdf(f) + (1 - y) * norm.logcdf(-f))
    else:
        raise ValueError(f"Unknown likelihood: {likelihood}")
    log_prior = -0.5 * np.dot(nu, nu)
    return float(log_lik + log_prior)


def log_posterior_batch(
    coords: np.ndarray,
    factor: np.ndarray,
    y: np.ndarray,
    likelihood: str,
) -> np.ndarray:
    """Vectorized log posterior over shape (nwalkers, dim)."""
    f = coords @ factor.T
    if likelihood == "sigmoid":
        p = expit(f)
        log_lik = np.sum(y[None, :] * np.log(p + 1e-10), axis=1)
        log_lik += np.sum((1 - y)[None, :] * np.log(1 - p + 1e-10), axis=1)
    elif likelihood == "probit":
        log_lik = np.sum(y[None, :] * norm.logcdf(f), axis=1)
        log_lik += np.sum((1 - y)[None, :] * norm.logcdf(-f), axis=1)
    else:
        raise ValueError(f"Unknown likelihood: {likelihood}")
    log_prior = -0.5 * np.sum(coords * coords, axis=1)
    return log_lik + log_prior


def grad_log_posterior_batch(
    coords: np.ndarray,
    factor: np.ndarray,
    y: np.ndarray,
    likelihood: str,
) -> np.ndarray:
    """Vectorized gradient wrt nu for shape (nwalkers, dim)."""
    f = coords @ factor.T
    if likelihood == "sigmoid":
        p = expit(f)
        grad_f = y[None, :] - p
    elif likelihood == "probit":
        imr_pos = np.exp(norm.logpdf(f) - norm.logcdf(f))
        imr_neg = np.exp(norm.logpdf(-f) - norm.logcdf(-f))
        grad_f = y[None, :] * imr_pos - (1 - y)[None, :] * imr_neg
    else:
        raise ValueError(f"Unknown likelihood: {likelihood}")
    return grad_f @ factor - coords


def grad_log_posterior(
    nu: np.ndarray,
    factor: np.ndarray,
    y: np.ndarray,
    likelihood: str,
) -> np.ndarray:
    """Gradient wrt a single latent vector nu."""
    f = factor @ nu
    if likelihood == "sigmoid":
        p = expit(f)
        grad_f = y - p
    elif likelihood == "probit":
        imr_pos = np.exp(norm.logpdf(f) - norm.logcdf(f))
        imr_neg = np.exp(norm.logpdf(-f) - norm.logcdf(-f))
        grad_f = y * imr_pos - (1 - y) * imr_neg
    else:
        raise ValueError(f"Unknown likelihood: {likelihood}")
    return factor.T @ grad_f - nu


def compute_tau_emcee(chain: np.ndarray) -> float:
    """
    Estimate integrated autocorrelation time using emcee's implementation.

    Expects an array shaped either (nsteps,) or (nsteps, nwalkers).
    """
    try:
        tau = emcee.autocorr.integrated_time(chain, quiet=True)
    except Exception as err:
        print(f"  WARNING: emcee tau estimate failed: {err}")
        return float("nan")

    tau = np.asarray(tau, dtype=float).reshape(-1)
    if tau.size == 0:
        return float("nan")
    return float(np.nanmean(tau))


def compute_ess_from_tau(n_steps: int, n_walkers: int, tau: float) -> float:
    """Convert tau to an approximate ensemble ESS."""
    if not np.isfinite(tau) or tau <= 0:
        return float("nan")
    return float((n_steps * n_walkers) / tau)


def make_mala_move(
    factor: np.ndarray,
    y: np.ndarray,
    step_size: float,
    likelihood: str,
) -> emcee.moves.MHMove:
    """Create an emcee MH move that uses a MALA proposal."""

    def proposal_function(coords: np.ndarray, random) -> tuple[np.ndarray, np.ndarray]:
        grads = grad_log_posterior_batch(coords, factor, y, likelihood)
        noise = random.randn(*coords.shape)
        drift = 0.5 * (step_size**2) * grads
        proposals = coords + drift + step_size * noise

        prop_grads = grad_log_posterior_batch(proposals, factor, y, likelihood)

        forward_diff = proposals - coords - drift
        reverse_diff = coords - proposals - 0.5 * (step_size**2) * prop_grads

        # emcee expects log q(x | x') - log q(x' | x).
        forward = -0.5 * np.sum(forward_diff * forward_diff, axis=1) / (step_size**2)
        reverse = -0.5 * np.sum(reverse_diff * reverse_diff, axis=1) / (step_size**2)
        factors = reverse - forward
        return proposals, factors

    return emcee.moves.MHMove(proposal_function)


def run_emcee_sampler(
    factor: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    n_warmup: int,
    n_walkers: int,
    seed: int,
    move,
    likelihood: str,
    init_scale: float = 0.1,
) -> dict:
    """Run emcee and return timing/statistics."""
    rng = np.random.RandomState(seed)
    dim = factor.shape[1]
    initial_state = init_scale * rng.randn(n_walkers, dim)

    sampler = emcee.EnsembleSampler(
        nwalkers=n_walkers,
        ndim=dim,
        log_prob_fn=log_posterior_batch,
        args=(factor, y, likelihood),
        moves=move,
        vectorize=True,
    )

    t0 = time.perf_counter()
    sampler.run_mcmc(initial_state, n_warmup, progress=False)
    warmup_time = time.perf_counter() - t0
    sampler.reset()

    t0 = time.perf_counter()
    sampler.run_mcmc(None, n_samples, progress=False)
    sample_time = time.perf_counter() - t0

    log_prob = sampler.get_log_prob()

    return {
        "per_step_time": float(sample_time / max(n_samples, 1)),
        "total_mcmc_time": float(sample_time),
        "warmup_time": float(warmup_time),
        "accept_rate": float(np.mean(sampler.acceptance_fraction)),
        "logp_trace": np.mean(log_prob, axis=1),
        "logp_by_walker": log_prob,
    }


def run_hmc(
    factor: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    n_warmup: int,
    seed: int,
    initial_step_size: float,
    n_leapfrog: int,
    likelihood: str,
    target_accept: float = 0.8,
    adapt_step_size: bool = True,
) -> dict:
    """Run a simple Euclidean HMC sampler with fixed mass matrix."""
    rng = np.random.default_rng(seed)
    dim = factor.shape[1]
    total_steps = n_warmup + n_samples

    nu = np.zeros(dim, dtype=float)
    logp = log_posterior(nu, factor, y, likelihood)

    step_times = np.zeros(total_steps, dtype=float)
    logp_trace = np.zeros(total_steps, dtype=float)
    accepts = np.zeros(total_steps, dtype=bool)

    step_size = initial_step_size
    log_step = np.log(initial_step_size)
    mu = np.log(10 * initial_step_size)
    log_step_bar = 0.0
    H_bar = 0.0
    gamma, t0_dual, kappa = 0.05, 10, 0.75

    for i in range(total_steps):
        t0 = time.perf_counter()
        if adapt_step_size and n_warmup > 0:
            if i < n_warmup:
                step_size = np.exp(log_step)
            elif i == n_warmup:
                step_size = np.exp(log_step_bar)

        current_nu = nu.copy()
        current_logp = logp
        momentum = rng.standard_normal(dim)

        grad = grad_log_posterior(current_nu, factor, y, likelihood)
        proposal_nu = current_nu.copy()
        proposal_p = momentum + 0.5 * step_size * grad

        for leapfrog_idx in range(n_leapfrog):
            proposal_nu = proposal_nu + step_size * proposal_p
            grad = grad_log_posterior(proposal_nu, factor, y, likelihood)
            if leapfrog_idx != n_leapfrog - 1:
                proposal_p = proposal_p + step_size * grad

        proposal_p = proposal_p + 0.5 * step_size * grad
        proposal_p = -proposal_p

        proposal_logp = log_posterior(proposal_nu, factor, y, likelihood)
        current_h = -current_logp + 0.5 * np.dot(momentum, momentum)
        proposal_h = -proposal_logp + 0.5 * np.dot(proposal_p, proposal_p)
        log_accept = current_h - proposal_h
        accept_prob = 1.0 if log_accept >= 0.0 else float(np.exp(log_accept))

        if np.log(rng.random()) < log_accept:
            nu = proposal_nu
            logp = proposal_logp
            accepts[i] = True

        if adapt_step_size and i < n_warmup:
            eta = 1.0 / (i + 1 + t0_dual)
            H_bar = (1 - eta) * H_bar + eta * (target_accept - accept_prob)
            log_step = mu - np.sqrt(i + 1) / gamma * H_bar
            eta_bar = (i + 1) ** (-kappa)
            log_step_bar = eta_bar * log_step + (1 - eta_bar) * log_step_bar

        step_times[i] = time.perf_counter() - t0
        logp_trace[i] = logp

    post = slice(n_warmup, total_steps)
    warmup = slice(0, n_warmup)
    warmup_time = float(np.sum(step_times[warmup]))
    per_step_time = float(np.mean(step_times[post]))
    total_mcmc_time = float(np.sum(step_times[post]))
    accept_rate = float(np.mean(accepts[post]))

    return {
        "per_step_time": per_step_time,
        "warmup_time": warmup_time,
        "total_mcmc_time": total_mcmc_time,
        "accept_rate": accept_rate,
        "logp_trace": logp_trace,
        "step_size": float(step_size),
        "n_leapfrog": int(n_leapfrog),
    }


def warn_if_acceptance_outside_band(
    sampler: str,
    accept_rate: float,
    lower: float,
    upper: float,
) -> None:
    if lower <= accept_rate <= upper:
        return
    direction = "below" if accept_rate < lower else "above"
    step_hint = "too large" if accept_rate < lower else "too small"
    print(
        f"  WARNING: {sampler} acceptance {accept_rate:.3f} {direction} target band "
        f"[{lower:.2f}, {upper:.2f}] - step size likely {step_hint}"
    )


def main() -> None:
    args = parse_args()
    data_dir = Path(args.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    n_samples = args.n_samples
    n_warmup = args.n_warmup
    n_seed_repeats = args.n_seed_repeats
    k_values = args.k_values
    likelihood = args.likelihood
    run_started_utc = datetime.datetime.now(datetime.timezone.utc)
    run_timestamp_utc = run_started_utc.isoformat()
    output_timestamp = run_started_utc.strftime("%Y%m%d-%H%M%S")
    hostname = socket.gethostname()
    job_id = os.environ.get("SLURM_JOB_ID", "not_set")

    print(f"Likelihood: {likelihood}")
    print(f"Running {n_seed_repeats} seed repeats per sampler/k configuration.")

    factor_dir = Path(args.factor_dir)
    y = np.load(factor_dir / "labels.npy").astype(np.float64)
    n_train = int(y.shape[0])

    results = []

    def result_row(
        *,
        sampler: str,
        k: int,
        n_walkers: int,
        seed: int,
        step_size: float,
        n_leapfrog: int | str,
        ess_logp: float,
        ess_per_sec: float,
        tau: float,
        accept_rate: float,
        factor_load_time_sec: float,
        warmup_time_sec: float,
        sampling_time_sec: float,
        per_step_time_sec: float,
        total_mcmc_time_sec: float,
    ) -> dict[str, object]:
        return {
            "experiment": "exp1c",
            "dataset": args.dataset_name,
            "factor_dir": str(factor_dir),
            "likelihood": likelihood,
            "n_train": n_train,
            "sampler": sampler,
            "k": int(k),
            "n_walkers": int(n_walkers),
            "seed": int(seed),
            "step_size": float(step_size),
            "n_leapfrog": n_leapfrog,
            "n_samples": int(n_samples),
            "n_warmup": int(n_warmup),
            "ess_logp": float(ess_logp),
            "ess_per_sec": float(ess_per_sec),
            "tau": float(tau),
            "accept_rate": float(accept_rate),
            "factor_load_time_sec": float(factor_load_time_sec),
            "warmup_time_sec": float(warmup_time_sec),
            "sampling_time_sec": float(sampling_time_sec),
            "per_step_time_sec": float(per_step_time_sec),
            "total_mcmc_time_sec": float(total_mcmc_time_sec),
            "hmc_target_accept": float(args.hmc_target_accept),
            "hmc_adapt_step_size": bool(args.hmc_adapt_step_size),
            "run_timestamp_utc": run_timestamp_utc,
            "hostname": hostname,
            "job_id": job_id,
        }

    for seed_offset in range(n_seed_repeats):
        print(f"\n=== Seed repeat {seed_offset + 1}/{n_seed_repeats} ===")
        for k in k_values:
            factor_path = factor_dir / f"factor_k{k}.npy"
            if not factor_path.exists():
                print(f"WARNING: {factor_path} not found, skipping k={k}")
                continue

            factor_start = time.perf_counter()
            F = np.load(factor_path).astype(np.float64)
            factor_load_time = time.perf_counter() - factor_start
            dim = F.shape[1]
            n_walkers = min(max(2 * dim + 2, 24), 100)

            actual_rank = max(dim, 1)
            rwm_step = args.rwm_step_constant / (actual_rank**0.5)
            gaussian_step = rwm_step**2
            mala_step = args.mala_step_constant / (actual_rank ** (1 / 3))
            hmc_step = args.hmc_step_constant / (actual_rank**0.25)
            hmc_leapfrog = 12

            rwm_seed = 1000 + k + seed_offset * 10000
            mala_seed = 2000 + k + seed_offset * 10000
            hmc_seed = 3000 + k + seed_offset * 10000

            print(
                f"k={k} actual_rank={dim} seed_offset={seed_offset}: "
                f"RWM step={rwm_step:.4f}, MALA step={mala_step:.4f}, "
                f"HMC step={hmc_step:.4f}"
            )

            # emcee's GaussianMove is exactly a Gaussian random-walk MH proposal here.
            gaussian_move = emcee.moves.GaussianMove(cov=gaussian_step, mode="vector")
            mala_move = make_mala_move(F, y, step_size=mala_step, likelihood=likelihood)

            gaussian_stats = run_emcee_sampler(
                F,
                y,
                n_samples=n_samples,
                n_warmup=n_warmup,
                n_walkers=n_walkers,
                seed=rwm_seed,
                move=gaussian_move,
                likelihood=likelihood,
            )
            mala_stats = run_emcee_sampler(
                F,
                y,
                n_samples=n_samples,
                n_warmup=n_warmup,
                n_walkers=n_walkers,
                seed=mala_seed,
                move=mala_move,
                likelihood=likelihood,
            )
            hmc_stats = run_hmc(
                F,
                y,
                n_samples=n_samples,
                n_warmup=n_warmup,
                seed=hmc_seed,
                initial_step_size=hmc_step,
                n_leapfrog=hmc_leapfrog,
                likelihood=likelihood,
                target_accept=args.hmc_target_accept,
                adapt_step_size=args.hmc_adapt_step_size,
            )

            tau_gaussian = compute_tau_emcee(gaussian_stats["logp_by_walker"])
            tau_mala = compute_tau_emcee(mala_stats["logp_by_walker"])
            tau_hmc = compute_tau_emcee(hmc_stats["logp_trace"][..., None])
            ess_gaussian = compute_ess_from_tau(n_samples, n_walkers, tau_gaussian)
            ess_mala = compute_ess_from_tau(n_samples, n_walkers, tau_mala)
            ess_hmc = compute_ess_from_tau(n_samples, 1, tau_hmc)

            essps_gaussian = ess_gaussian / max(
                gaussian_stats["total_mcmc_time"], 1e-12
            )
            essps_mala = ess_mala / max(mala_stats["total_mcmc_time"], 1e-12)
            essps_hmc = ess_hmc / max(hmc_stats["total_mcmc_time"], 1e-12)

            results.append(
                result_row(
                    sampler="RWM",
                    k=dim,
                    n_walkers=n_walkers,
                    seed=rwm_seed,
                    step_size=rwm_step,
                    n_leapfrog="",
                    ess_logp=ess_gaussian,
                    ess_per_sec=essps_gaussian,
                    tau=tau_gaussian,
                    accept_rate=gaussian_stats["accept_rate"],
                    factor_load_time_sec=factor_load_time,
                    warmup_time_sec=gaussian_stats["warmup_time"],
                    sampling_time_sec=gaussian_stats["total_mcmc_time"],
                    per_step_time_sec=gaussian_stats["per_step_time"],
                    total_mcmc_time_sec=gaussian_stats["total_mcmc_time"],
                )
            )
            results.append(
                result_row(
                    sampler="MALA",
                    k=dim,
                    n_walkers=n_walkers,
                    seed=mala_seed,
                    step_size=mala_step,
                    n_leapfrog="",
                    ess_logp=ess_mala,
                    ess_per_sec=essps_mala,
                    tau=tau_mala,
                    accept_rate=mala_stats["accept_rate"],
                    factor_load_time_sec=factor_load_time,
                    warmup_time_sec=mala_stats["warmup_time"],
                    sampling_time_sec=mala_stats["total_mcmc_time"],
                    per_step_time_sec=mala_stats["per_step_time"],
                    total_mcmc_time_sec=mala_stats["total_mcmc_time"],
                )
            )
            results.append(
                result_row(
                    sampler="HMC",
                    k=dim,
                    n_walkers=1,
                    seed=hmc_seed,
                    step_size=hmc_stats["step_size"],
                    n_leapfrog=hmc_leapfrog,
                    ess_logp=ess_hmc,
                    ess_per_sec=essps_hmc,
                    tau=tau_hmc,
                    accept_rate=hmc_stats["accept_rate"],
                    factor_load_time_sec=factor_load_time,
                    warmup_time_sec=hmc_stats["warmup_time"],
                    sampling_time_sec=hmc_stats["total_mcmc_time"],
                    per_step_time_sec=hmc_stats["per_step_time"],
                    total_mcmc_time_sec=hmc_stats["total_mcmc_time"],
                )
            )

            warn_if_acceptance_outside_band(
                "RWM", gaussian_stats["accept_rate"], lower=0.15, upper=0.35
            )
            warn_if_acceptance_outside_band(
                "MALA", mala_stats["accept_rate"], lower=0.45, upper=0.70
            )
            warn_if_acceptance_outside_band(
                "HMC", hmc_stats["accept_rate"], lower=0.70, upper=0.90
            )

    available_k_values = sorted({int(r["k"]) for r in results})
    if not available_k_values:
        print("No factor files were loaded successfully; exiting without CSV output.")
        return

    fmt = "{:<8} {:<9} {:>7} {:>8} {:>10} {:>8} {:>12} {:>10} {:>10} {:>8}"
    for k in available_k_values:
        print(f"k={k}")
        print(
            fmt.format(
                "Sampler",
                "Likelihood",
                "Seed",
                "Walkers",
                "Step size",
                "Accept",
                "Per-step(s)",
                "ESS",
                "ESS/sec",
                "tau",
            )
        )
        for row in [r for r in results if int(r["k"]) == k]:
            print(
                fmt.format(
                    row["sampler"],
                    row["likelihood"],
                    f"{row['seed']}",
                    f"{row['n_walkers']}",
                    f"{row['step_size']:.4f}",
                    f"{row['accept_rate']:.3f}",
                    f"{row['per_step_time_sec']:.6f}",
                    f"{row['ess_logp']:.1f}",
                    f"{row['ess_per_sec']:.2f}",
                    f"{row['tau']:.2f}",
                )
            )
        print()

    print("Mean +/- std ESS/sec across seed repeats:")
    for k in available_k_values:
        parts = []
        for sampler in ["RWM", "MALA", "HMC"]:
            values = [
                float(r["ess_per_sec"])
                for r in results
                if int(r["k"]) == k and r["sampler"] == sampler
            ]
            if values:
                arr = np.asarray(values, dtype=float)
                if sampler == "RWM":
                    parts.append(
                        f"{sampler} ess/sec = {np.mean(arr):.1f} +/- {np.std(arr):.1f}"
                    )
                else:
                    parts.append(f"{sampler} = {np.mean(arr):.1f} +/- {np.std(arr):.1f}")
        print(f"k={k}: " + ", ".join(parts))

    csv_columns = [
        "experiment",
        "dataset",
        "factor_dir",
        "likelihood",
        "n_train",
        "sampler",
        "k",
        "n_walkers",
        "seed",
        "step_size",
        "n_leapfrog",
        "n_samples",
        "n_warmup",
        "ess_logp",
        "ess_per_sec",
        "tau",
        "accept_rate",
        "factor_load_time_sec",
        "warmup_time_sec",
        "sampling_time_sec",
        "per_step_time_sec",
        "total_mcmc_time_sec",
        "hmc_target_accept",
        "hmc_adapt_step_size",
        "run_timestamp_utc",
        "hostname",
        "job_id",
    ]
    csv_path = (
        data_dir
        / f"exp1c_emcee_real_data_{likelihood}_{output_timestamp}.csv"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved CSV to {csv_path.resolve()}")


if __name__ == "__main__":
    main()
