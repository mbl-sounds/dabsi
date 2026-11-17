#!/bin/bash -l
# One-time setup on a wice login node, after cloning the repo to $VSC_DATA/dabsi.
# Data is not in git: copy it first from the local machine (-L follows the local symlinks):
#   rsync -aL .librispeech .myriad <user>@<wice-login>:$VSC_DATA/dabsi/work/
set -euo pipefail
CODE=${CODE:-$VSC_DATA/dabsi}
VENV="$VSC_DATA/dabsi/venv-$VSC_ARCH_LOCAL"

module purge
module load Python/3.14.2-GCCcore-15.2.0

cd "$CODE"
python -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip

# Dependency pins are read from pyproject.toml so the cluster and local environments cannot drift.
REQS=$(python - <<'EOF'
import tomllib

deps = tomllib.load(open("pyproject.toml", "rb"))["tool"]["poetry"]["dependencies"]
for name, spec in deps.items():
    if name != "python":
        print(name + spec)
EOF
)
# shellcheck disable=SC2086  # intentional word splitting: specs contain no spaces
python -m pip install $REQS

# Compiles lib/*_cy.pyx for this machine; the local macOS .so files cannot be used here.
python setup.py build_ext --inplace

python -c "import lib.centralized_cy, lib.distgrad_cy, lib.dsaawet_cy, lib.simulation_utils_cy, dbsi.experiments; print('setup ok')"
