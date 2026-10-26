#!/bin/bash -l
#SBATCH --clusters=wice
#SBATCH --partition=batch_sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=24:00:00
#
# One-time parameter tuning on one node. Writes $CODE/tuned/<experiment>.json.
#   sbatch --account=<project> --export=ALL,EXPERIMENT=wgn_rooms slurm/tune.slurm --where SNR=20,50
# Extra arguments after the script name are passed to tune.py.
# Copy tuned/<experiment>.json back to the repo and commit it before running the full experiment.

set -euo pipefail
: "${EXPERIMENT:?set EXPERIMENT=<name>, e.g. --export=ALL,EXPERIMENT=wgn_rooms}"
CODE=${CODE:-$VSC_DATA/dabsi}
WORK=${WORK:-$VSC_DATA/dabsi/work}  # cwd: .librispeech and .myriad are resolved relative to it
OUT=${OUT:-$VSC_DATA/dabsi/data}

module purge
module load Python/3.14.2-GCCcore-15.2.0
source "$VSC_DATA/dabsi/venv-$VSC_ARCH_LOCAL/bin/activate"

mkdir -p "$WORK"
cd "$WORK"
python "$CODE/tune.py" "$EXPERIMENT" --out "$OUT" \
    --processes "$SLURM_CPUS_PER_TASK" "$@"
