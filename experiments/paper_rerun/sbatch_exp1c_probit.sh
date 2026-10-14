#!/bin/bash
#SBATCH --job-name=exp1c_probit
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9
#SBATCH --output=slurm_logs/exp1c_probit_%A_%a.out
#SBATCH --error=slurm_logs/exp1c_probit_%A_%a.err
#SBATCH --partition=a100_short
#SBATCH --exclude=a100-4032

# Controlled SVGP (inducing points + lengthscale frozen to the RPCholesky
# pivots/bandwidth for this seed), probit likelihood -- gpytorch's
# BernoulliLikelihood, the default for run_exp1_controlled.py and the same
# link run_exp1.py's standard SVGP uses. Sibling of sbatch_exp1c_logistic.sh,
# which runs the same model with --likelihood logistic instead.
# 5 seeds x 2 datasets = 10 array tasks, one (dataset, seed) per task.
#
# Pivots are rebuilt from the seed rather than read from a saved exp2
# posterior (--pivots-file), so this does not need to wait on the exp2 array:
# run_exp1_controlled.py's rebuild uses the identical RNG sequence as
# run_exp2.py and reproduces the same pivots/bandwidth (~20s extra on CPU).

DATASETS=(pcam pcam pcam pcam pcam camelyon17 camelyon17 camelyon17 camelyon17 camelyon17)
SEEDS=(0 1 2 3 4 0 1 2 3 4)

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

SCRATCH="/gpfs/scratch/sd6701"
REPO_DIR="$SCRATCH/personal/gpc-mcmc-cholesky-factorization"
DATA_ROOT="$SCRATCH/gpc-mcmc-cholesky-factorization/experiments/ML-Final-Project/Datasets"

if [ -z "${RUN_ID:-}" ]; then
  RUN_ID="batch-${SLURM_ARRAY_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
  echo "WARNING: RUN_ID not supplied; defaulting to ${RUN_ID}." >&2
fi
BATCH_DIR="$REPO_DIR/data/paper_rerun/$RUN_ID"
RUN_DIR="$BATCH_DIR/exp1c_probit/runs"

cd "$REPO_DIR"
mkdir -p slurm_logs "$RUN_DIR"

eval "$(conda shell.bash hook)"
conda activate /gpfs/scratch/sd6701/conda-envs/gpc

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/experiments:$REPO_DIR/experiments/paper_rerun:$REPO_DIR:$PYTHONPATH"

echo "=== Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID: exp1c_probit dataset=$DATASET seed=$SEED ==="
echo "Node: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "run id: $RUN_ID   run dir: $RUN_DIR"
echo "Started at $(date)"

if [ "$DATASET" = "camelyon17" ]; then
  EVAL_SPLITS="test valid"
else
  EVAL_SPLITS="test"
fi

python experiments/paper_rerun/run_exp1_controlled.py \
    --dataset "$DATASET" \
    --data-root "$DATA_ROOT" \
    --seed "$SEED" \
    --likelihood probit \
    --eval-splits $EVAL_SPLITS \
    --run-dir "$RUN_DIR"

echo "=== Finished at $(date) ==="
