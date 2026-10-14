#!/bin/bash
#SBATCH --job-name=exp1c_logistic
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9
#SBATCH --output=slurm_logs/exp1c_logistic_%A_%a.out
#SBATCH --error=slurm_logs/exp1c_logistic_%A_%a.err
#SBATCH --partition=a100_short
#SBATCH --exclude=a100-4032

# 5 seeds x 2 datasets = 10 jobs
DATASETS=(pcam pcam pcam pcam pcam camelyon17 camelyon17 camelyon17 camelyon17 camelyon17)
SEEDS=(0 1 2 3 4 0 1 2 3 4)

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

SCRATCH="/gpfs/scratch/sd6701"
REPO_DIR="$SCRATCH/personal/gpc-mcmc-cholesky-factorization"
DATA_ROOT="$SCRATCH/gpc-mcmc-cholesky-factorization/experiments/ML-Final-Project/Datasets"

# RUN_ID groups this script with sbatch_exp1.sh, sbatch_exp1c_probit.sh and
# sbatch_exp2.sh into one batch directory; submit_all.sh sets it. Falls back
# to a timestamp so this script still works submitted standalone. Previously
# this wrote to a flat data/paper_rerun/runs, which would collide with
# sbatch_exp1c_probit.sh's seed*.json files (both are the "exp1c" experiment
# group) -- each likelihood variant now gets its own run-dir.
if [ -z "${RUN_ID:-}" ]; then
  RUN_ID="batch-${SLURM_ARRAY_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
  echo "WARNING: RUN_ID not supplied; defaulting to ${RUN_ID}." >&2
fi
BATCH_DIR="$REPO_DIR/data/paper_rerun/$RUN_ID"
RUN_DIR="$BATCH_DIR/exp1c_logistic/runs"

cd "$REPO_DIR"
mkdir -p slurm_logs "$RUN_DIR"

eval "$(conda shell.bash hook)"
conda activate /gpfs/scratch/sd6701/conda-envs/gpc

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/experiments:$REPO_DIR/experiments/paper_rerun:$REPO_DIR:$PYTHONPATH"

echo "=== Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID: dataset=$DATASET seed=$SEED ==="
echo "Node: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "run id: $RUN_ID   run dir: $RUN_DIR"
echo "Started at $(date)"

python experiments/paper_rerun/run_exp1_controlled.py \
    --dataset "$DATASET" \
    --data-root "$DATA_ROOT" \
    --seed "$SEED" \
    --eval-splits test valid \
    --likelihood logistic \
    --run-dir "$RUN_DIR"

echo "=== Finished at $(date) ==="
