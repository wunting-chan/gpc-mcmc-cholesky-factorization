#!/bin/bash
# Run this from: /gpfs/scratch/sd6701/personal/gpc-mcmc-cholesky-factorization
# Usage: bash scripts/verify_results.sh

RUNDIR="data/paper_rerun/runs"

echo "=========================================="
echo "CONTROLLED SVGP LOGISTIC (Job 1 / exp1c)"
echo "=========================================="
echo ""
echo "--- Checking what result files exist ---"
find "$RUNDIR" -name "*.json" | grep -i "exp1c\|controlled" | sort
echo ""
echo "--- Checking what CSV files exist ---"
find "$RUNDIR" -name "*.csv" | sort
echo ""

echo "--- All JSON files in runs directory ---"
find "$RUNDIR" -name "*.json" | sort
echo ""

echo "=========================================="
echo "COMPARING CAMELYON17 TEST RESULTS"
echo "=========================================="

echo ""
echo "--- exp1c logistic (controlled SVGP) files ---"
for f in $(find "$RUNDIR" -name "*.json" | grep -i "exp1c" | grep -i "camelyon17" | grep -v "valid" | sort); do
    echo "FILE: $f"
    python3 -c "
import json
with open('$f') as fh:
    d = json.load(fh)
print(f'  record_id: {d.get(\"record_id\",\"?\")}')
print(f'  accuracy:  {d.get(\"accuracy\",\"?\")}')
print(f'  auroc:     {d.get(\"auroc\",\"?\")}')
print(f'  ece:       {d.get(\"ece\",\"?\")}')
print(f'  fnr:       {d.get(\"false_negative_rate\",\"?\")}')
print(f'  brier:     {d.get(\"brier\",\"?\")}')
print(f'  nll:       {d.get(\"negative_log_likelihood_mean\", d.get(\"nll\",\"?\"))}')
print(f'  seed:      {d.get(\"seed\",\"?\")}')
print(f'  likelihood:{d.get(\"likelihood\",\"?\")}')
print(f'  bandwidth: {d.get(\"kernel_bandwidth\",\"?\")}')
print()
"
done

echo ""
echo "--- exp2 HMC (rerun) files ---"
for f in $(find "$RUNDIR" -name "*.json" | grep "exp2" | grep -i "camelyon17" | grep -v "valid" | sort); do
    echo "FILE: $f"
    python3 -c "
import json
with open('$f') as fh:
    d = json.load(fh)
print(f'  record_id: {d.get(\"record_id\",\"?\")}')
print(f'  accuracy:  {d.get(\"accuracy\",\"?\")}')
print(f'  auroc:     {d.get(\"auroc\",\"?\")}')
print(f'  ece:       {d.get(\"ece\",\"?\")}')
print(f'  fnr:       {d.get(\"false_negative_rate\",\"?\")}')
print(f'  brier:     {d.get(\"brier\",\"?\")}')
print(f'  nll:       {d.get(\"negative_log_likelihood_mean\", d.get(\"nll\",\"?\"))}')
print(f'  seed:      {d.get(\"seed\",\"?\")}')
print(f'  tau_logp:  {d.get(\"tau_logp\",\"?\")}')
print(f'  likelihood:{d.get(\"likelihood\",\"?\")}')
print(f'  bandwidth: {d.get(\"kernel_bandwidth\",\"?\")}')
print()
"
done

echo ""
echo "=========================================="
echo "CHECK: did exp1c_logistic CSV come from the right JSON files?"
echo "=========================================="
echo ""
echo "Looking for any CSV that was generated..."
find "$RUNDIR" -name "*.csv" -newer "$RUNDIR" -exec echo "CSV: {}" \; -exec head -2 {} \; 2>/dev/null
find . -maxdepth 2 -name "*logistic*.csv" -exec echo "CSV: {}" \; -exec head -3 {} \; 2>/dev/null
find . -maxdepth 2 -name "*exp1c*.csv" -exec echo "CSV: {}" \; -exec head -3 {} \; 2>/dev/null

echo ""
echo "=========================================="
echo "CHECK: slurm logs for exp1c_logistic"
echo "=========================================="
echo ""
ls -lt slurm_logs/exp1c_logistic_*.out 2>/dev/null | head -10
echo ""
echo "--- Last 5 lines of each camelyon17 test log ---"
for f in slurm_logs/exp1c_logistic_*.out; do
    echo "=== $f ==="
    tail -5 "$f" 2>/dev/null
    echo ""
done
