#!/bin/bash -l
#SBATCH --clusters=wice
#SBATCH --partition=batch_sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#
# One array task = one node = one shard of one experiment. Array size = number of nodes.
#   sbatch --account=<project> --export=ALL,EXPERIMENT=wgn_rooms slurm/wice.slurm
# Results go to $VSC_DATA/dabsi/data/<experiment>; resubmitting resumes the open jobs.
# Set --time from a pilot run (see the sec and peak_rss_mb fields in the npz metadata).

set -euo pipefail
: "${EXPERIMENT:?set EXPERIMENT=<name>, e.g. --export=ALL,EXPERIMENT=wgn_rooms}"
CODE=${CODE:-$VSC_DATA/dabsi/src}
WORK=${WORK:-$VSC_DATA/dabsi/work}  # cwd: .librispeech and .myriad are resolved relative to it
OUT=${OUT:-$VSC_DATA/dabsi/data}

module purge
module load <python-module>  # placeholder: a Python 3.12-3.14 module from `module av Python`
source "$VSC_DATA/dabsi/venv-$VSC_ARCH_LOCAL/bin/activate"

mkdir -p "$WORK"
cd "$WORK"
python "$CODE/run_experiments.py" "$EXPERIMENT" --out "$OUT" \
    --processes "$SLURM_CPUS_PER_TASK" \
    --shard "$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT"
