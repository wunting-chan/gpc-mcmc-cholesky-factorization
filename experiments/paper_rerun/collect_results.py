"""
Collect every per-seed JSON row written by this batch's four sbatch scripts
(sbatch_exp1.sh, sbatch_exp1c_probit.sh, sbatch_exp1c_logistic.sh,
sbatch_exp2.sh) into one CSV, with sanity checks.

Layout each script writes into (see submit_all.sh):

    <batch-dir>/exp1/runs/<dataset>/exp1/seed00N.json
    <batch-dir>/exp1c_probit/runs/<dataset>/exp1c/seed00N.json
    <batch-dir>/exp1c_logistic/runs/<dataset>/exp1c/seed00N.json
    <batch-dir>/exp2/runs/<dataset>/exp2/seed00N.json

(plus "*_valid" experiment directories for camelyon17, which evaluates on both
test and valid splits; pcam evaluates on test only).

Both exp1c variants write the "exp1c" experiment name (schema.py bakes the
likelihood into model_architecture / embedding_variant, not into the group
name), so they would collide in one directory -- that is exactly why each
variant has its own <batch-dir>/<variant>/runs/ subtree. "variant" (the first
path component under <batch-dir>) is what actually distinguishes them here.

This does not average seeds (scripts/paper_rerun_aggregate.py does that, on
its own flat run-dir layout). This is a flat, traceable dump of every row
plus a pass/fail check, meant to catch a broken or partial batch before
anyone trusts an average.

Usage:
    python collect_results.py --batch-dir data/paper_rerun/<RUN_ID> \
        [--out <path>] [--expect-seeds 0 1 2 3 4]

Exit code is 0 if every check passes, 1 otherwise -- submit_all.sh's final
job depends on the four array jobs via --dependency=afterany, so this is
what actually tells you whether the batch is clean.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import schema  # noqa: E402

EXTRA_COLUMNS = ["variant", "group", "likelihood_tag", "source_path", "row_issues"]
CSV_COLUMNS = schema.COLUMNS + EXTRA_COLUMNS

# (variant dir name) -> likelihood the model_architecture / embedding_variant
# suffix is expected to carry, for variants where that is fixed by the script
# rather than chosen per row. exp1 and exp2 have no --likelihood flag at all.
EXPECTED_LIKELIHOOD = {
    "exp1c_probit": "probit",
    "exp1c_logistic": "logistic",
}

RATE_COLUMNS = ["accuracy", "auroc", "auprc", "ece", "brier"]
NONNEGATIVE_COLUMNS = ["n_train", "n_test", "tp", "fp", "tn", "fn"]

# Per-dataset expected (variant, experiment-group) combos.
# CAMELYON17 evaluates on test + valid; PCam evaluates on test only.
_ALL_TEST = {
    ("exp1", "exp1"),
    ("exp1c_probit", "exp1c"),
    ("exp1c_logistic", "exp1c"),
    ("exp2", "exp2"),
}
_CAMELYON17_VALID = {
    ("exp1", "exp1_valid"),
    ("exp1c_probit", "exp1c_valid"),
    ("exp1c_logistic", "exp1c_valid"),
    ("exp2", "exp2_valid"),
}
EXPECTED_BY_DATASET = {
    "camelyon17": _ALL_TEST | _CAMELYON17_VALID,
    "pcam": _ALL_TEST,
}


def as_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, str):
        text = value.strip()
        if not text or text in (schema.NOT_APPLICABLE, schema.NOT_COMPUTED):
            return None
        try:
            number = float(text)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def likelihood_tag(row: dict) -> str:
    """Pull 'probit'/'logistic' off model_architecture or embedding_variant."""
    for field in ("model_architecture", "embedding_variant"):
        value = str(row.get(field, ""))
        for tag in ("probit", "logistic"):
            if value.endswith(tag):
                return tag
    return ""


def check_row(row: dict, variant: str, dataset: str, group: str, seed_from_name: int) -> list[str]:
    issues = []

    if row.get("dataset") != dataset:
        issues.append(f"dataset mismatch: path={dataset} row={row.get('dataset')}")
    if row.get("experiment") != group:
        issues.append(f"experiment mismatch: path={group} row={row.get('experiment')}")
    seed_val = as_number(row.get("seed"))
    if seed_val is None or int(seed_val) != seed_from_name:
        issues.append(f"seed mismatch: filename={seed_from_name} row={row.get('seed')}")

    expected_lik = EXPECTED_LIKELIHOOD.get(variant)
    if expected_lik is not None:
        found_lik = likelihood_tag(row)
        if found_lik != expected_lik:
            issues.append(
                f"likelihood mismatch: variant={variant} expected={expected_lik} "
                f"found={found_lik or '(none)'}"
            )

    for col in RATE_COLUMNS:
        val = as_number(row.get(col))
        if val is None:
            # Not every experiment populates every rate column (e.g. auprc).
            continue
        if not (0.0 <= val <= 1.0):
            issues.append(f"{col} out of [0,1]: {val}")

    for col in NONNEGATIVE_COLUMNS:
        val = as_number(row.get(col))
        if val is None:
            continue
        if val < 0:
            issues.append(f"{col} negative: {val}")

    tp, fp, tn, fn = (as_number(row.get(c)) for c in ("tp", "fp", "tn", "fn"))
    n_test = as_number(row.get("n_test"))
    if None not in (tp, fp, tn, fn, n_test):
        total = tp + fp + tn + fn
        if abs(total - n_test) > 0.5:
            issues.append(f"tp+fp+tn+fn={total:.0f} != n_test={n_test:.0f}")

    nll = as_number(row.get("negative_log_likelihood_mean"))
    if nll is not None and nll < 0:
        issues.append(f"negative_log_likelihood_mean negative: {nll}")

    if variant == "exp2":
        tau = as_number(row.get("tau_logp"))
        if tau is not None and tau < 0:
            issues.append(f"tau_logp negative (chain too short?): {tau}")

    return issues


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-dir", required=True,
                   help="e.g. data/paper_rerun/<RUN_ID> (the directory holding "
                        "exp1/, exp1c_probit/, exp1c_logistic/, exp2/).")
    p.add_argument("--out", default=None,
                   help="Output CSV path. Default: <batch-dir>/collected_results.csv")
    p.add_argument("--expect-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="Seeds every (variant, dataset, experiment) group must have "
                        "exactly once. Default: 0 1 2 3 4.")
    args = p.parse_args()

    batch_dir = Path(args.batch_dir)
    if not batch_dir.is_dir():
        raise SystemExit(f"No such batch directory: {batch_dir}")
    out_path = Path(args.out) if args.out else batch_dir / "collected_results.csv"

    # <batch-dir>/<variant>/runs/<dataset>/<experiment>/seedNNN.json
    paths = sorted(batch_dir.glob("*/runs/*/*/seed*.json"))

    all_rows = []
    groups = defaultdict(list)  # (variant, dataset, group) -> [seed, ...]
    hard_failures = 0

    for path in paths:
        rel = path.relative_to(batch_dir)
        variant, _runs, dataset, group, filename = rel.parts
        try:
            seed_from_name = int(filename[len("seed"):-len(".json")])
        except ValueError:
            print(f"SKIP (unparseable filename): {path}", file=sys.stderr)
            continue

        with open(path, encoding="utf-8") as handle:
            row = json.load(handle)

        issues = check_row(row, variant, dataset, group, seed_from_name)
        if issues:
            hard_failures += len(issues)

        out_row = {col: row.get(col, schema.NOT_APPLICABLE) for col in schema.COLUMNS}
        out_row.update({
            "variant": variant,
            "group": group,
            "likelihood_tag": likelihood_tag(row),
            "source_path": str(rel),
            "row_issues": "; ".join(issues),
        })
        all_rows.append(out_row)
        groups[(variant, dataset, group)].append(seed_from_name)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"Wrote {len(all_rows)} rows to {out_path}")

    # --- group-level checks: duplicate / missing seeds ------------------
    expected = set(args.expect_seeds)
    group_problems = []
    for (variant, dataset, group), seeds in sorted(groups.items()):
        seen = set()
        dupes = set()
        for s in seeds:
            (dupes if s in seen else seen).add(s)
        missing = expected - set(seeds)
        label = f"{variant}/{dataset}/{group}"
        if dupes:
            group_problems.append(f"{label}: duplicate seed(s) {sorted(dupes)}")
        if missing:
            group_problems.append(f"{label}: missing seed(s) {sorted(missing)} "
                                   f"(have {sorted(set(seeds))})")
        extra = set(seeds) - expected
        if extra:
            group_problems.append(f"{label}: unexpected seed(s) {sorted(extra)}")

    for dataset in sorted({d for (_v, d, _g) in groups}):
        expected_groups = EXPECTED_BY_DATASET.get(dataset, _ALL_TEST)
        for variant, group in sorted(expected_groups):
            if (variant, dataset, group) not in groups:
                group_problems.append(f"{variant}/{dataset}/{group}: no runs found at all")

    print(f"\n{len(all_rows)} rows across {len(groups)} (variant, dataset, experiment) groups.")
    print(f"Rows with at least one issue: {sum(1 for r in all_rows if r['row_issues'])}")
    print(f"Total issue count (rows): {hard_failures}")

    if group_problems:
        print("\nGROUP-LEVEL PROBLEMS:")
        for problem in group_problems:
            print(f"  - {problem}")
    else:
        print("\nNo group-level problems (seed sets complete, no duplicates).")

    print("\nPer-group summary (mean of available numeric columns):")
    for (variant, dataset, group), seeds in sorted(groups.items()):
        rows = [r for r in all_rows
                if r["variant"] == variant and r["dataset"] == dataset and r["group"] == group]
        accs = [as_number(r.get("accuracy")) for r in rows]
        aurocs = [as_number(r.get("auroc")) for r in rows]
        accs = [a for a in accs if a is not None]
        aurocs = [a for a in aurocs if a is not None]
        acc_str = f"{sum(accs) / len(accs):.4f}" if accs else "n/a"
        auroc_str = f"{sum(aurocs) / len(aurocs):.4f}" if aurocs else "n/a"
        print(f"  {variant:16s} {dataset:11s} {group:11s} n={len(rows):<2d} "
              f"accuracy={acc_str:8s} auroc={auroc_str}")

    ok = hard_failures == 0 and not group_problems
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
