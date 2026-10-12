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

# 5 seeds x 2 datasets = 10 jobs
DATASETS=(pcam pcam pcam pcam pcam camelyon17 camelyon17 camelyon17 camelyon17 camelyon17)
SEEDS=(0 1 2 3 4 0 1 2 3 4)

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

SCRATCH="/gpfs/scratch/sd6701"
REPO_DIR="$SCRATCH/personal/gpc-mcmc-cholesky-factorization"
DATA_ROOT="$SCRATCH/gpc-mcmc-cholesky-factorization/experiments/ML-Final-Project/Datasets"

cd "$REPO_DIR"
mkdir -p slurm_logs data/paper_rerun/runs

source activate base

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/experiments:$REPO_DIR/experiments/paper_rerun:$REPO_DIR:$PYTHONPATH"

echo "=== Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID: dataset=$DATASET seed=$SEED ==="
echo "Node: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "Started at $(date)"

python experiments/paper_rerun/run_exp1_controlled.py \
    --dataset "$DATASET" \
    --data-root "$DATA_ROOT" \
    --seed "$SEED" \
    --eval-splits test valid \
    --likelihood logistic \
    --run-dir data/paper_rerun/runs

echo "=== Finished at $(date) ==="
