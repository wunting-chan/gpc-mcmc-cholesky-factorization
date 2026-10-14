#!/bin/bash
#SBATCH --job-name=exp2
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9
#SBATCH --output=slurm_logs/exp2_%A_%a.out
#SBATCH --error=slurm_logs/exp2_%A_%a.err
#SBATCH --partition=a100_short

# RPCholesky factor + low-rank HMC (exp2). HMC's target uses a logistic link
# unconditionally -- run_exp2.py has no --likelihood flag, this is not a
# choice. Pure NumPy/SciPy, no GPU is requested or used (same as
# sbatch_exp2_cam17_rerun.sh, which this is modeled on); the a100_short
# partition is used only because that reference script already proved it
# schedules and runs this CPU-only job fine (job 28388453, ~20min/task).
# 5 seeds x 2 datasets = 10 array tasks, one (dataset, seed) per task.

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
RUN_DIR="$BATCH_DIR/exp2/runs"

cd "$REPO_DIR"
mkdir -p slurm_logs "$RUN_DIR"

eval "$(conda shell.bash hook)"
conda activate /gpfs/scratch/sd6701/conda-envs/gpc

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/experiments:$REPO_DIR/experiments/paper_rerun:$REPO_DIR:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "=== Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID: exp2 dataset=$DATASET seed=$SEED ==="
echo "Node: $(hostname)"
echo "run id: $RUN_ID   run dir: $RUN_DIR"
echo "Started at $(date)"

python experiments/paper_rerun/run_exp2.py \
    --dataset "$DATASET" \
    --data-root "$DATA_ROOT" \
    --seed "$SEED" \
    --run-dir "$RUN_DIR"

echo "=== Finished at $(date) ==="
