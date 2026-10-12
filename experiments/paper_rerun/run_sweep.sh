#!/bin/bash
# Submit the leapfrog sweep job to SLURM
# Usage: bash experiments/paper_rerun/run_sweep.sh

cd "$(dirname "$0")"
mkdir -p slurm_logs

echo "Submitting sweep_leapfrog_cam17 to SLURM..."
sbatch sweep_leapfrog_cam17.sbatch

echo "Check status with: squeue -u $USER"
echo "View output with:  tail -f slurm_logs/sweep_lf_cam17_<jobid>.out"
