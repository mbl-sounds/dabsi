"""Prior sweeps: final NPM over lambda for simulated and measured rooms.

Usage: python plot_prior_sweep.py .data/prior_mismatch_sweep_long .data/measured_prior_sweep

Figures go to ``--plots`` as prior_sweeps_<grid>.pdf: one row per experiment, one column per SNR.
Tables go to ``--tables``: <experiment>_<grid>.tex (median final NPM [dB], IQR in brackets)
and <experiment>_summary.csv (all statistics per grid point and algorithm).
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from dbsi import analysis as an

ROOT = Path(__file__).resolve().parent
ALG = re.compile(r"dabsi_([a-z]+)(?:_l([0-9.e+-]+))?$")
# prior, label, colour, marker; the no-prior run is a reference line
SIMULATED = [
    ("matched", "Matched", "tab:blue", "o"),
    ("slight", "Slight", "tab:green", "s"),
    ("distinct", "Distinct", "tab:red", "^"),
    ("lowpass", "Low-pass 250 Hz", "tab:purple", "d"),
]
# RT60 matched prior (2.1 s) and the default simulated prior (0.32 s)
MEASURED = [
    ("matched", "RT60 matched (2.1 s)", "tab:blue", "o"),
    ("default", "Default (0.32 s)", "tab:orange", "v"),
]
# experiment folder -> (row title, priors)
EXPERIMENTS = {
    "prior_mismatch_sweep_long": ("Simulated rooms", SIMULATED),
    "measured_prior_sweep": ("Measured SAL RIRs", MEASURED),
}


def tagged(summ: pl.DataFrame, priors) -> pl.DataFrame:
    """Summary rows with ``prior`` and ``lambd`` parsed from the algorithm name (no prior: lambd 0)."""
    known = {"noprior"} | {p for p, *_ in priors}
    rows = []
    for r in summ.to_dicts():
        m = ALG.match(r["alg"])
        if m and m.group(1) in known:
            rows.append({**r, "prior": m.group(1), "lambd": float(m.group(2) or 0.0)})
    return pl.DataFrame(rows)


def point(g: pl.DataFrame, prior: str, snr) -> pl.DataFrame:
    return g.filter((pl.col("prior") == prior) & (pl.col("SNR") == snr)).sort("lambd")


def fmt(k, v) -> str:
    return f"{k}{v:g}" if isinstance(v, (int, float)) else f"{k}{v}"


def figure(rows, snrs):
    """One row per experiment (title, priors, summary of the grid point), one column per SNR."""
    fig, axes = plt.subplots(
        len(rows),
        len(snrs),
        figsize=(400 / 72.27, 2.3 * len(rows)),
        sharey="row",
        squeeze=False,
    )
    letters = iter("abcdefgh")
    for r, (title, priors, g) in enumerate(rows):
        entries = {}  # legend label -> handle, collected over the row
        for c, snr in enumerate(snrs):
            ax = axes[r][c]
            ref = point(g, "noprior", snr)
            if ref.height:
                ax.axhline(ref["median"][0], color="k", ls=":", lw=1, label="No prior")
            for prior, name, color, marker in priors:
                s = point(g, prior, snr)
                if s.height == 0:
                    continue
                med = s["median"].to_numpy()
                ax.errorbar(
                    s["lambd"].to_numpy(),
                    med,
                    yerr=[med - s["q1"].to_numpy(), s["q3"].to_numpy() - med],
                    color=color,
                    linestyle="-",
                    marker=marker,
                    markersize=5,
                    markerfacecolor="white",
                    capsize=2,
                    lw=1,
                    label=name,
                )
            h, lab = ax.get_legend_handles_labels()
            entries.update(zip(lab, h))
            ax.set_xscale("log")
            ax.set_xlabel(r"Regularization $\lambda_0$")
            ax.text(
                0.03,
                0.95,
                rf"({next(letters)}) {title}, SNR ${snr:g}$ dB",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )
            ax.tick_params(axis="both", which="major", labelsize=8, length=2.5, pad=2)
            ax.tick_params(axis="both", which="minor", length=1.5)
            ax.grid(True, which="major", alpha=0.25, linewidth=0.45)
            ax.grid(True, which="minor", alpha=0.10, linewidth=0.35)
            ax.margins(x=0.02)
            if c > 0:
                ax.tick_params(axis="y", which="both", labelleft=False)
        axes[r][0].set_ylim(top=0)
        axes[r][0].set_ylabel("Final NPM [dB]", labelpad=1)
        axes[r][0].legend(
            list(entries.values()),
            list(entries.keys()),
            loc="lower left",
            bbox_to_anchor=(0, 1.02),
            ncol=len(entries),
            frameon=False,
            fontsize=8,
            handlelength=1.8,
            columnspacing=1.0,
        )
    fig.subplots_adjust(
        left=0.11, right=0.985, bottom=0.1, top=0.9, wspace=0.05, hspace=0.5
    )
    return fig


def cell(g: pl.DataFrame, prior: str, snr, lam: float) -> str:
    lam = 0.0 if prior == "noprior" else lam
    r = g.filter(
        (pl.col("prior") == prior) & (pl.col("SNR") == snr) & (pl.col("lambd") == lam)
    )
    if r.height == 0:
        return "--"
    med, q1, q3 = r["median"][0], r["q1"][0], r["q3"][0]
    return f"{med:.1f} [{(q3 - q1) / 2:.1f}]"


def table(g: pl.DataFrame, snrs, lambdas, title: str, priors) -> str:
    table_rows = [("noprior", "No prior")] + [(p, n) for p, n, *_ in priors]
    blocks = []
    for snr in snrs:
        lines = [
            f"% {title}, SNR {snr:g} dB: median final NPM [dB], IQR half-width in brackets",
            rf"\begin{{tabular}}{{l{'c' * len(lambdas)}}}",
            r"\hline",
            r"Prior & " + " & ".join(f"{lam:g}" for lam in lambdas) + r" \\",
            r"\hline",
        ]
        for prior, name in table_rows:
            cells = [cell(g, prior, snr, lam) for lam in lambdas]
            lines.append(f"{name} & " + " & ".join(cells) + r" \\")
        lines += [r"\hline", r"\end{tabular}", ""]
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "exp_dirs", nargs="+", help="experiment folders written by run_experiments.py"
    )
    p.add_argument(
        "--tail", type=int, default=50, help="frames averaged for the final NPM"
    )
    p.add_argument("--plots", default=str(ROOT / "plots"))
    p.add_argument("--tables", default=str(ROOT / "tables"))
    a = p.parse_args()

    plots, tables = Path(a.plots), Path(a.tables)
    plots.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": False,
            "pgf.rcfonts": False,
            "font.size": 9,
        }
    )
    summaries = {}  # experiment name -> (summary, grid axes)
    for d in a.exp_dirs:
        name = Path(d).name
        if name not in EXPERIMENTS:
            raise SystemExit(
                f"unknown experiment {name!r}; expected one of {sorted(EXPERIMENTS)}"
            )
        df = an.load(d)
        summ = tagged(an.summary(df, "npm", tail=a.tail), EXPERIMENTS[name][1])
        if summ.height == 0:
            raise SystemExit(f"no prior algorithm names found in {name}")
        summ.write_csv(tables / f"{name}_summary.csv")

        failed = int((summ["n_total"] - summ["n_valid"]).sum())
        print(
            f"{name}: runs per algorithm and SNR {summ['n_total'].min()}..{summ['n_total'].max()}, "
            f"invalid runs: {failed}, algorithms: {summ['alg'].n_unique()}"
        )
        summaries[name] = (summ, [k for k in an.params(df) if k != "SNR"])

    if len({tuple(keys) for _, keys in summaries.values()}) > 1:
        raise SystemExit("experiments have different grid axes")
    keys = next(iter(summaries.values()))[1]
    snrs = sorted(
        {s for summ, _ in summaries.values() for s in summ["SNR"].unique().to_list()}
    )
    lambdas = sorted(
        {
            v
            for summ, _ in summaries.values()
            for v in summ["lambd"].unique().to_list()
            if v > 0
        }
    )
    by_point = {
        name: {pt: g for pt, g in summ.group_by(keys, maintain_order=True)}
        for name, (summ, _) in summaries.items()
    }
    for pt in sorted({pt for groups in by_point.values() for pt in groups}):
        point_name = "_".join(fmt(k, v) for k, v in zip(keys, pt))
        title = " ".join(f"{k}={v}" for k, v in zip(keys, pt))
        present = [n for n in by_point if pt in by_point[n]]
        fig = figure(
            [(EXPERIMENTS[n][0], EXPERIMENTS[n][1], by_point[n][pt]) for n in present],
            snrs,
        )
        fig.savefig(plots / f"prior_sweeps_{point_name}.pdf", bbox_inches="tight")
        plt.close(fig)
        for n in present:
            (tables / f"{n}_{point_name}.tex").write_text(
                table(by_point[n][pt], snrs, lambdas, title, EXPERIMENTS[n][1])
            )
        print(
            f"wrote prior_sweeps_{point_name}: experiments {present}, lambda {lambdas}, SNR {snrs}"
        )


if __name__ == "__main__":
    main()
