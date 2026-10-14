#!/bin/bash
# Submit the full rerun batch: exp1, exp1c_probit, exp1c_logistic, exp2 --
# each 5 seeds x 2 datasets (10 array tasks) -- then collect_results.py as a
# dependent job that runs once all four arrays have finished (success or
# not: afterany, not afterok, so a partial/failed array still gets collected
# and reported on rather than leaving the batch silently uncollected).
#
# Usage:
#   ./submit_all.sh                  # fresh RUN_ID, e.g. full_rerun-20261013T...
#   RUN_ID=my-batch ./submit_all.sh  # reuse/name a specific batch dir
set -euo pipefail

SCRATCH="/gpfs/scratch/sd6701"
REPO_DIR="$SCRATCH/personal/gpc-mcmc-cholesky-factorization"
cd "$REPO_DIR"

RUN_ID="${RUN_ID:-full_rerun-$(date -u +%Y%m%dT%H%M%SZ)}"
BATCH_DIR="$REPO_DIR/data/paper_rerun/$RUN_ID"

mkdir -p slurm_logs "$BATCH_DIR"

echo "RUN_ID    : $RUN_ID"
echo "BATCH_DIR : $BATCH_DIR"
echo ""

JOB_EXP1=$(sbatch --parsable --export=ALL,RUN_ID="$RUN_ID" experiments/paper_rerun/sbatch_exp1.sh)
echo "exp1            -> job $JOB_EXP1  (array 0-9)"

JOB_EXP1C_PROBIT=$(sbatch --parsable --export=ALL,RUN_ID="$RUN_ID" experiments/paper_rerun/sbatch_exp1c_probit.sh)
echo "exp1c_probit    -> job $JOB_EXP1C_PROBIT  (array 0-9)"

JOB_EXP1C_LOGISTIC=$(sbatch --parsable --export=ALL,RUN_ID="$RUN_ID" experiments/paper_rerun/sbatch_exp1c_logistic.sh)
echo "exp1c_logistic  -> job $JOB_EXP1C_LOGISTIC  (array 0-9)"

JOB_EXP2=$(sbatch --parsable --export=ALL,RUN_ID="$RUN_ID" experiments/paper_rerun/sbatch_exp2.sh)
echo "exp2            -> job $JOB_EXP2  (array 0-9)"

DEPS="afterany:${JOB_EXP1}:${JOB_EXP1C_PROBIT}:${JOB_EXP1C_LOGISTIC}:${JOB_EXP2}"

COLLECT_CMD="cd '$REPO_DIR'; eval \"\$(conda shell.bash hook)\"; conda activate /gpfs/scratch/sd6701/conda-envs/gpc; export PYTHONPATH='$REPO_DIR/src:$REPO_DIR/experiments:$REPO_DIR/experiments/paper_rerun:$REPO_DIR:\$PYTHONPATH'; python experiments/paper_rerun/collect_results.py --batch-dir '$BATCH_DIR'"

JOB_COLLECT=$(sbatch --parsable \
  --job-name=collect_results \
  --partition=cpu_short \
  --time=00:30:00 \
  --mem=8G \
  --cpus-per-task=1 \
  --dependency="$DEPS" \
  --output="slurm_logs/collect_results_${RUN_ID}_%j.out" \
  --error="slurm_logs/collect_results_${RUN_ID}_%j.err" \
  --wrap="$COLLECT_CMD")
echo "collect_results -> job $JOB_COLLECT  (depends on $DEPS)"

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "Results land under:"
echo "  $BATCH_DIR/{exp1,exp1c_probit,exp1c_logistic,exp2}/runs/<dataset>/<experiment>/seed*.json"
echo "  $BATCH_DIR/collected_results.csv   (after job $JOB_COLLECT)"
