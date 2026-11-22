"""Summarize and plot the dsaawet_sweep results."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dbsi import analysis as an  # noqa: E402

COLS = [
    "signal",
    "alg",
    "n_valid",
    "n_total",
    "median",
    "q1",
    "q3",
    "conv_frame",
    "fail_rate",
    "scale_spread_db",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("exp_dir", nargs="?", default=".data/dsaawet_sweep")
    args = p.parse_args()
    exp = Path(args.exp_dir)

    df = an.load(str(exp))
    s = an.summary(df)
    print(s.sort(["signal", "alg"]).select(COLS))

    algs = sorted(df["alg"].unique().to_list())
    titles = {"white": "speech-like WGN", "stationary": "stationary WGN"}
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
    for ax, sig in zip(axes, ["white", "stationary"]):
        for a in algs:
            c = an.curves(df, signal=sig, alg=a)  # [R, T] in dB
            ax.plot(np.median(c, axis=0), label=a)
        ax.set_title(titles[sig])
        ax.set_xlabel("frame")
        ax.set_ylim(top=0)
    axes[0].set_ylabel("misalignment [dB]")
    axes[0].legend()

    out = exp / "dsaawet_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
