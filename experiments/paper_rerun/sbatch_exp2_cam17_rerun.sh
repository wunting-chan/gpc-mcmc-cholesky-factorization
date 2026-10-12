#!/bin/bash
#SBATCH --job-name=exp2_cam17_rerun
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH --output=slurm_logs/exp2_cam17_rerun_%A_%a.out
#SBATCH --error=slurm_logs/exp2_cam17_rerun_%A_%a.err
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:a100:1

SEED=$SLURM_ARRAY_TASK_ID

REPO_DIR="$HOME/personal/gpc-mcmc-cholesky-factorization"
DATA_ROOT="$HOME/gpc-mcmc-cholesky-factorization/experiments/ML-Final-Project/Datasets"

cd "$REPO_DIR"
mkdir -p slurm_logs data/paper_rerun/runs

source activate base

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/experiments:$REPO_DIR/experiments/paper_rerun:$REPO_DIR:$PYTHONPATH"

echo "=== Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID: camelyon17 seed=$SEED ==="
echo "Node: $(hostname)"
echo "Started at $(date)"

python experiments/paper_rerun/run_exp2.py \
    --dataset camelyon17 \
    --data-root "$DATA_ROOT" \
    --seed "$SEED" \
    --eval-splits test valid \
    --run-dir data/paper_rerun/runs

echo "=== Finished at $(date) ==="
