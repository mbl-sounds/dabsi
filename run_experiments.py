"""Run an experiment:  python run_experiments.py wgn_rooms --out .data --processes 8

Resumable: finished jobs (files in <out>/<experiment>/) are skipped. Add --dry-run to list the open jobs.
Cluster: --shard i/n runs every n-th open job; one job-array task per node (see slurm/wice.slurm).
"""

import os

# One thread per process: parallelism comes from the process pool, not from BLAS/OpenMP.
# Must be set before numpy/torch are imported (spawned workers inherit it).
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402


def main():
    from dbsi.experiments import EXPERIMENTS
    from dbsi.runner import default_processes, run_experiment

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", choices=sorted(EXPERIMENTS))
    p.add_argument("--out", default=".data")
    p.add_argument("--processes", type=int, default=default_processes())
    p.add_argument("--shard", help="i/n: run only jobs k with k %% n == i")
    p.add_argument("--runs", type=int, help="override number of Monte Carlo runs")
    p.add_argument("--seed", type=int, help="override seed")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--force", action="store_true", help="overwrite manifest of a different config"
    )
    p.add_argument(
        "--no-tuned", action="store_true", help="ignore tuned/<experiment>.json"
    )
    a = p.parse_args()

    exp = dict(EXPERIMENTS[a.experiment])
    tuned = Path(__file__).resolve().parent / "tuned" / f"{a.experiment}.json"
    if tuned.exists() and not a.no_tuned:
        from dbsi.tuning import apply_tuned

        print(f"using tuned parameters from {tuned}", flush=True)
        exp = apply_tuned(exp, tuned)
    if a.runs is not None:
        exp["runs"] = a.runs
    if a.seed is not None:
        exp["seed"] = a.seed
    shard = tuple(int(x) for x in a.shard.split("/")) if a.shard else None
    sys.exit(
        1 if run_experiment(exp, a.out, a.processes, a.dry_run, a.force, shard) else 0
    )


if __name__ == "__main__":
    main()
