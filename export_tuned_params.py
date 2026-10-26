"""Write the tuned parameters into the LaTeX parameter table of the paper:

    python export_tuned_params.py

Reads ``tuned/<experiment>.json`` (written by ``tune.py``) for the WGN-RIR, simulated-RIR and
measured-RIR experiments, applies them to the experiment configs exactly like ``run_experiments.py``
does, and writes ``tables/parameters_tuned.tex``. The values are read from
the resolved configs, so the table cannot drift from the values used in the runs. Do not edit the
generated file by hand.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TUNED_DIR = HERE / "tuned"
OUT = HERE / "tables" / "parameters_tuned.tex"

# (column header, experiment name, DABSI variant that is shown in the paper)
COLUMNS = [
    ("\\gls{wgn} \\glspl{rir}", "wgn_rooms", "dabsi"),
    ("Simulated \\glspl{rir}", "simulated_rooms", "dabsi_l1e-05"),
    ("Measured \\glspl{rir}", "measured_rooms", "dabsi_l1e-05"),
]


def _fmt(v) -> str:
    """Number wrapped in inline math, e.g. 0.4 -> \\(0.4\\), 1e-5 -> \\(10^{-5}\\)."""
    return f"\\({_math(v)}\\)"


def _math(v) -> str:
    v = float(v)
    if v == 0:
        return "0"
    if 1e-2 <= abs(v) < 1e2:
        s = f"{v:g}"
        return s + ".0" if "." not in s and "e" not in s else s
    mant, exp = f"{v:.3e}".split("e")
    if float(mant) == 1:
        return f"10^{{{int(exp)}}}"
    return f"{float(mant):g}\\cdot 10^{{{int(exp)}}}"


def _resolved(name: str) -> dict:
    """Experiment config with the tuned parameters applied (as in run_experiments.py)."""
    from dbsi.experiments import EXPERIMENTS
    from dbsi.tuning import apply_tuned

    path = TUNED_DIR / f"{name}.json"
    if not path.exists():
        sys.exit(f"missing {path}: run `python tune.py {name} --where SNR=20,50` first")
    return apply_tuned(EXPERIMENTS[name], str(path))


NA = "--"  # not applicable to this column

# Constants hard-coded in dbsi/harness.py and lib/simulation_utils.py (not part of the configs)
VAD_THRESHOLD_DB = -60
VAD_SMOOTHING = 10
WALL_MARGIN_M = 0.1
RIR_PERTURBATION_DB = {"room": 80, "meas": 20}  # RIR noise below the channel power
PARTITION_AMPLITUDE = "\\(\\mathcal{U}([0.5,1])\\)"
RIR_SOURCE = {"wgn": "i.i.d. Gaussian", "room": "image source", "meas": "MYRiAD"}


def _int(v) -> str:
    return f"\\({int(v)}\\)"


def _join(values) -> str:
    return f"\\({', '.join(str(int(v)) for v in values)}\\)"


def _group(title: str) -> str:
    return "        \\hline \\multicolumn{4}{|l|}{\\textit{" + title + "}} \\\\\n"


def _row(label: str, cells: list, rule: str = "\\hline") -> str:
    return f"        {rule} {label} & " + " & ".join(cells) + " \\\\\n"


def main():
    from dbsi.config import DABSIConfig

    resolved = {name: _resolved(name) for _, name, _ in COLUMNS}
    commits = sorted(
        {
            str(json.loads((TUNED_DIR / f"{name}.json").read_text()).get("git"))
            for _, name, _ in COLUMNS
        }
    )
    bfgs_tol = DABSIConfig.__dataclass_fields__["bfgs_tol"].default

    def per_col(fn) -> list:
        """One cell per column from ``fn(experiment, variant)``; None means not applicable."""
        return [
            NA if (value := fn(resolved[name], variant)) is None else value
            for _, name, variant in COLUMNS
        ]

    def dabsi(key: str) -> list:
        return per_col(lambda e, v: _fmt(e["algs"][v][key]))

    def baseline(alg: str, key: str) -> list:
        # baseline names carry the same lambda tag as the DABSI variant ("" or "_l1e-05")
        return per_col(lambda e, v: _fmt(e["algs"][alg + v[len("dabsi") :]][key]))

    def pair(a: str, b: str, key: str) -> list:
        """Value of two baselines in one cell each, shown as 'a / b' if they differ."""
        return [
            x if x == y else f"{x} / {y}"
            for x, y in zip(baseline(a, key), baseline(b, key))
        ]

    def prior_of(e: dict, v: str):
        return e["algs"][v].get("prior") if e["algs"][v]["lambd"] else None

    def rt60_cell(rt60, e: dict) -> str:
        # None means 5 L / f_s, the default of the room generator
        return _fmt(5 * e["L"] / e["scenario"]["fs"] if rt60 is None else rt60)

    def uniform(size) -> str:
        return f"\\(\\mathcal{{U}}([{size[0]:g},{size[1]:g}])\\)"

    def is_room(e: dict) -> bool:
        return e["scenario"]["kind"] == "room"

    rows = [
        _group("Signal and network"),
        _row("Filter length \\(L\\) [samples]", per_col(lambda e, v: _int(e["L"]))),
        _row("Excitation length [samples]", per_col(lambda e, v: _int(e["scenario"]["nr_samples"]))),
        _row(
            "Sampling rate \\(f_s\\) [Hz]",
            per_col(lambda e, v: _int(e["scenario"]["fs"]) if "fs" in e["scenario"] else None),
        ),
        _row("Partitions", per_col(lambda e, v: _int(e["scenario"]["partitions"]))),
        _row("Partition amplitude", per_col(lambda e, v: PARTITION_AMPLITUDE)),
        _row("Nodes \\(M\\)", per_col(lambda e, v: _join(e["grid"]["M"]))),
        _row("Receive neighbourhood \\(N_i\\) (incl. self)", per_col(lambda e, v: _join(e["grid"]["N_i"]))),
        _row("Graph", per_col(lambda e, v: "directed, random")),
        _row("SNR [dB]", per_col(lambda e, v: f"\\({int(min(e['grid']['SNR']))}\\)--\\({int(max(e['grid']['SNR']))}\\)")),
        _row("Noise", per_col(lambda e, v: "white Gaussian" if e["scenario"]["kind"] == "wgn" else "random PSD")),
        _row("VAD threshold [dB]", per_col(lambda e, v: _int(VAD_THRESHOLD_DB))),
        _row("VAD smoothing [samples]", per_col(lambda e, v: _int(VAD_SMOOTHING))),
        _group("Room impulse responses"),
        _row("RIR generation", per_col(lambda e, v: RIR_SOURCE[e["scenario"]["kind"]])),
        _row("Room size [m]", per_col(lambda e, v: uniform(e["scenario"]["roomsize"]) if is_room(e) else None)),
        _row("Wall margin [m]", per_col(lambda e, v: _fmt(WALL_MARGIN_M) if is_room(e) else None)),
        _row(
            "RIR perturbation [dB below channel]",
            per_col(lambda e, v: _int(RIR_PERTURBATION_DB[e["scenario"]["kind"]]) if e["scenario"]["kind"] in RIR_PERTURBATION_DB else None),
        ),
        _row("\\(\\text{RT}_{60}\\) [s]", per_col(lambda e, v: rt60_cell(e["scenario"].get("rt60"), e) if is_room(e) else None)),
        _row("Prior rooms \\(N_{\\text{rooms}}\\)", per_col(lambda e, v: _int(prior_of(e, v)["n_rooms"]) if prior_of(e, v) else None)),
        _row("Prior room size [m]", per_col(lambda e, v: uniform(prior_of(e, v)["roomsize"]) if prior_of(e, v) else None)),
        _row("Prior \\(\\text{RT}_{60}\\) [s]", per_col(lambda e, v: rt60_cell(prior_of(e, v)["rt60"], e) if prior_of(e, v) else None)),
        _group("DABSI"),
        _row("\\(\\rho\\)", dabsi("rho")),
        _row("\\(\\mu\\)", dabsi("mu")),
        _row("\\(\\eta\\)", dabsi("eta")),
        _row("\\(\\lambda_0\\)", dabsi("lambd")),
        _row("Local iterations", per_col(lambda e, v: _int(e["algs"][v]["local_iters"]))),
        _row("Distributed averaging", per_col(lambda e, v: "yes" if e["algs"][v]["da"] else "no")),
        _row("GEV regularisation \\(\\varepsilon_{\\text{GEV}}\\)", dabsi("gev_reg")),
        _row("GEV clipping \\(c_{\\text{GEV}}\\)", per_col(lambda e, v: _int(e["algs"][v]["gev_lim"]))),
        _row("BFGS tolerance", [_fmt(bfgs_tol)] * len(COLUMNS)),
        _group("Compared methods"),
        _row("\\(\\mu_{\\text{MCQN}}\\)", baseline("mcqn", "mu")),
        _row("\\(\\mu_{\\text{S-MCQN}}\\)", baseline("spmcqn", "mu")),
        _row("\\(\\eta\\) (MCQN / S-MCQN)", pair("mcqn", "spmcqn", "eta")),
        _row("\\(\\lambda\\) (MCQN / S-MCQN)", pair("mcqn", "spmcqn", "lambd")),
        _row("\\(\\alpha_{\\text{DGD}}\\)", baseline("dgd", "alpha")),
        _row("\\(\\alpha_{\\text{DSAAWET}}\\)", baseline("dsaawet", "alpha")),
        _group("Simulation"),
        _row("Monte Carlo runs", per_col(lambda e, v: _int(e["runs"]))),
        _row("Random seed", per_col(lambda e, v: _int(e["seed"]))),
        _row(
            "Frames per run",
            # common frame count of run_job in dbsi/experiments.py: mic length is nr_samples + L - 1
            per_col(lambda e, v: _int(len(range(0, e["scenario"]["nr_samples"] + e["L"] - 1 - 4 * e["L"], e["L"])))),
        ),
    ]

    header = " & ".join(f"\\textbf{{{h}}}" for h, _, _ in COLUMNS)
    lines = [
        "% Generated by export_tuned_params.py from tuned/*.json. Do not edit by hand.",
        f"% tuning git info: {'; '.join(commits)}",
        "\\begin{table*}",
        "    \\centering",
        "    \\caption{Simulation parameters for \\gls{dabsi} and the compared methods. Values are read from the resolved experiment configurations.}",
        "    \\label{tab:06-numerical-simulations:parameters}",
        "    \\def\\arraystretch{1.15}%",
        "    \\begin{tabular}{|p{0.30\\textwidth}|>{\\centering\\arraybackslash}p{0.2\\textwidth}|>{\\centering\\arraybackslash}p{0.2\\textwidth}|>{\\centering\\arraybackslash}p{0.2\\textwidth}|}",
        f"        \\hline \\textbf{{Parameter}} & {header} \\\\",
    ]
    tail = ["        \\hline", "    \\end{tabular}", "\\end{table*}", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n" + "".join(rows) + "\n".join(tail))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
