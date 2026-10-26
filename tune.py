"""One-time parameter tuning before a full experiment:
    python tune.py wgn_rooms --where SNR=20,50 --processes 8

Writes tuned/<experiment>.json (commit it); run_experiments.py then uses these parameters automatically.
Restartable: algorithms already tuned in that file are skipped (--force redoes everything).
"""

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

TUNED_DIR = Path(__file__).resolve().parent / "tuned"


def _where(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = [
            json.loads(x) if x.replace(".", "", 1).replace("-", "", 1).isdigit() else x
            for x in v.split(",")
        ]
    return out


def main():
    from dbsi.experiments import EXPERIMENTS
    from dbsi.runner import default_processes
    from dbsi.tuning import tune

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", choices=sorted(EXPERIMENTS))
    p.add_argument(
        "--where",
        nargs="*",
        help="grid points to tune on, e.g. SNR=20,50 N_i=2 (default: first value of each axis)",
    )
    p.add_argument("--algs", nargs="*", help="only these algorithm names")
    p.add_argument("--runs", type=int, default=3, help="tuning scenarios per grid point")
    p.add_argument("--min-frames", type=int, default=250)
    p.add_argument("--max-frames", type=int, default=4000)
    p.add_argument(
        "--conv-db",
        type=float,
        default=-10.0,
        help="stop early once the best candidate is below this",
    )
    p.add_argument("--out", default=".data")
    p.add_argument("--processes", type=int, default=default_processes())
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    exp = EXPERIMENTS[a.experiment]
    tune(
        exp,
        str(Path(a.out) / exp["name"]),
        str(TUNED_DIR / f"{exp['name']}.json"),
        _where(a.where),
        a.runs,
        a.min_frames,
        a.max_frames,
        a.conv_db,
        a.algs,
        a.processes,
        force=a.force,
    )


if __name__ == "__main__":
    main()
