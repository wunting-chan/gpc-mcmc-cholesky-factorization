#!/bin/bash
#SBATCH --job-name=sweep_lf_cam17
#SBATCH --output=slurm_logs/sweep_lf_cam17_%j.out
#SBATCH --error=slurm_logs/sweep_lf_cam17_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

REPO_DIR="$HOME/gpc-mcmc-cholesky-factorization"
DATA_ROOT="$REPO_DIR/experiments/ML-Final-Project/Datasets"
OUT_DIR="$REPO_DIR/experiments/paper_rerun/sweep_results"

cd "$REPO_DIR"
mkdir -p "$OUT_DIR"
mkdir -p experiments/paper_rerun/slurm_logs

source activate base

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/experiments:$REPO_DIR:$PYTHONPATH"

echo "=== Job $SLURM_JOB_ID started at $(date) ==="
echo "Node: $(hostname)"
echo "Data root: $DATA_ROOT"

python experiments/paper_rerun/sweep_leapfrog_cam17.py \
    --data-root "$DATA_ROOT" \
    --out-dir "$OUT_DIR" \
    --leapfrog-values 10 16 25 35 50

echo "=== Job finished at $(date) ==="
