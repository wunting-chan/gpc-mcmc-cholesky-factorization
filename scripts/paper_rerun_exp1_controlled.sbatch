#!/bin/bash
# Controlled SVGP (Experiment A): GPyTorch SVGP with inducing points frozen at
# the RPCholesky pivots, fixed lengthscale/outputscale, only q(u) trained.
# 5 seeds x 2 datasets, each array task = one seed on both datasets, scored on
# the test AND validation splits. GPU job (~5 min per dataset on an A100).
#
# Submit from the project root, sharing RUN_ID with the exp2 batch so the
# controlled runs land next to the HMC runs they are compared against:
#   RUN_ID="run-$(date -u +%Y%m%dT%H%M%SZ)"
#   sbatch --export=ALL,RUN_ID=$RUN_ID scripts/paper_rerun_exp1_controlled.sbatch
#
# Pivots: if the exp2 run for the same seed already wrote
#   <RUN_DIR>/<dataset>/exp2_seed<N>_posterior.npz
# it is used directly; otherwise the factor is rebuilt with the same seed,
# which gives identical pivots (~20 s on CPU). Set PIVOTS=rebuild to force a
# rebuild, or PIVOTS_RUN_ID=<other batch> to take pivots from another batch.
#
# Overridable via --export: PROJECT_ROOT, CONDA_ENV, CONDA_SH, DATA_ROOT,
# DATASETS ("pcam camelyon17"), EVAL_SPLITS ("test valid"), LIKELIHOOD
# (probit|logistic), LENGTHSCALE (override the per-seed bandwidth), EPOCHS.
#
# NOTE: verify the partition name against `sinfo -s` before submitting, or
# pass --partition=<name> (and --account=<acct> if required) on the sbatch
# command line; command-line flags override the directives below.
#SBATCH --job-name=paper_rerun_exp1c
#SBATCH --partition=gpu4_medium
#SBATCH --array=1-5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/paper_rerun/exp1c_%A_%a.out
#SBATCH --error=slurm_logs/paper_rerun/exp1c_%A_%a.err

set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/scratch/sd6701/gpc-mcmc-cholesky-factorization}"
CONDA_ENV="${CONDA_ENV:-/gpfs/scratch/sd6701/conda-envs/gpc}"
CONDA_SH="${CONDA_SH:-/gpfs/scratch/sd6701/miniconda3/etc/profile.d/conda.sh}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/experiments/ML-Final-Project/Datasets}"
DATASETS="${DATASETS:-pcam camelyon17}"
EVAL_SPLITS="${EVAL_SPLITS:-test valid}"
LIKELIHOOD="${LIKELIHOOD:-probit}"
EPOCHS="${EPOCHS:-50}"
PIVOTS="${PIVOTS:-auto}"

if [ -z "${RUN_ID:-}" ]; then
  RUN_ID="batch-${SLURM_ARRAY_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
  echo "WARNING: RUN_ID not supplied; defaulting to ${RUN_ID}." >&2
fi
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/data/paper_rerun/${RUN_ID}/runs}"
PIVOTS_RUN_DIR="${PROJECT_ROOT}/data/paper_rerun/${PIVOTS_RUN_ID:-${RUN_ID}}/runs"
SEED="${SLURM_ARRAY_TASK_ID:-1}"

EXTRA_ARGS="--likelihood ${LIKELIHOOD} --num-epochs ${EPOCHS}"
if [ -n "${LENGTHSCALE:-}" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --lengthscale ${LENGTHSCALE}"
fi
if [ -n "${MAX_TRAIN:-}" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --max-train ${MAX_TRAIN}"
fi

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs/paper_rerun "${RUN_DIR}"

# shellcheck disable=SC1090
source "${CONDA_SH}"
set +u
conda activate "${CONDA_ENV}"
set -u

PYTHON="${CONDA_PREFIX}/bin/python"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "=============================================================="
echo "exp1c controlled SVGP | seed=${SEED} | job=${SLURM_JOB_ID:-local}"
echo "node=$(hostname) | start=$(date)"
echo "run id : ${RUN_ID}   datasets: ${DATASETS}   splits: ${EVAL_SPLITS}"
echo "likelihood: ${LIKELIHOOD}   pivots: ${PIVOTS}"
echo "=============================================================="

"${PYTHON}" -c "import torch, gpytorch; print('torch', torch.__version__, 'gpytorch', gpytorch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true

for DATASET in ${DATASETS}; do
  PIVOT_ARGS=""
  POSTERIOR="${PIVOTS_RUN_DIR}/${DATASET}/exp2_seed${SEED}_posterior.npz"
  if [ "${PIVOTS}" != "rebuild" ] && [ -f "${POSTERIOR}" ]; then
    PIVOT_ARGS="--pivots-file ${POSTERIOR}"
    echo "using pivots from ${POSTERIOR}"
  else
    echo "no saved posterior for ${DATASET} seed ${SEED}; rebuilding pivots"
  fi
  echo ""
  echo "--- exp1c ${DATASET} seed ${SEED} ---"
  # shellcheck disable=SC2086
  "${PYTHON}" experiments/paper_rerun/run_exp1_controlled.py \
    --dataset "${DATASET}" \
    --data-root "${DATA_ROOT}" \
    --seed "${SEED}" \
    --run-dir "${RUN_DIR}" \
    --eval-splits ${EVAL_SPLITS} \
    ${PIVOT_ARGS} ${EXTRA_ARGS}
done

echo ""
echo "Done: $(date)"
