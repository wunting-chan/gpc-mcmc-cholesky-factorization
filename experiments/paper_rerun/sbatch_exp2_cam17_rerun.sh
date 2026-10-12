#!/bin/bash
#SBATCH --job-name=exp2_cam17_rerun
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH --output=slurm_logs/exp2_cam17_rerun_%A_%a.out
#SBATCH --error=slurm_logs/exp2_cam17_rerun_%A_%a.err
#SBATCH --partition=a100_short


SEED=$SLURM_ARRAY_TASK_ID

SCRATCH="/gpfs/scratch/sd6701"
REPO_DIR="$SCRATCH/personal/gpc-mcmc-cholesky-factorization"
DATA_ROOT="$SCRATCH/gpc-mcmc-cholesky-factorization/experiments/ML-Final-Project/Datasets"

cd "$REPO_DIR"
mkdir -p slurm_logs data/paper_rerun/runs

eval "$(conda shell.bash hook)"
conda activate /gpfs/scratch/sd6701/conda-envs/gpc

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
