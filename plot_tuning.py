"""Plot the successive-halving history of tuned/<experiment>.json:
    python plot_tuning.py tuned/prior_mismatch.json --out figures/tuning

One figure per grid point and algorithm: left, score [dB] of every candidate against the number of frames;
right, the dB trace of the candidates of the last round (the longest runs). Diverged values are not drawn.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_history(tuned_path, out_dir) -> list[Path]:
    tuned = json.loads(Path(tuned_path).read_text())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, (pk, algs) in enumerate(tuned["history"].items()):
        for alg, rounds in algs.items():
            if not rounds:
                continue
            series = {}
            for r in rounds:
                for cand, score in r["scores"].items():
                    if math.isfinite(score):
                        series.setdefault(cand, []).append((r["frames"], score))

            metric = rounds[0].get("metric", "npm").upper()
            fig, (ax, ax_tr) = plt.subplots(1, 2, figsize=(11, 4))
            for cand, pts in series.items():
                frames, scores = zip(*pts)
                ax.plot(frames, scores, marker="o", label=cand)
            ax.axhline(tuned["settings"]["conv_db"], color="gray", ls="--", lw=1)
            ax.set_xlabel("frames")
            ax.set_ylabel(f"score [dB] (mean {metric}, second half)")
            ax.legend(fontsize=6)

            last = rounds[-1]
            for cand, trace in last.get("traces", {}).items():
                xs = [(k + 0.5) / len(trace) * last["frames"] for k in range(len(trace))]
                pts = [(x, y) for x, y in zip(xs, trace) if math.isfinite(y)]
                if pts:
                    x, y = zip(*pts)
                    ax_tr.plot(x, y, marker="o", ms=3, label=cand)
            ax_tr.axhline(tuned["settings"]["conv_db"], color="gray", ls="--", lw=1)
            ax_tr.set_xlabel("frames (block centre)")
            ax_tr.set_ylabel(f"block mean [dB] ({metric})")
            ax_tr.set_title(f"last round: {last['frames']} frames", fontsize=8)
            if ax_tr.lines:
                ax_tr.legend(fontsize=6)

            chosen = tuned["params"].get(pk, {}).get(alg)
            fig.suptitle(f"{pk}\n{alg} -> {chosen}", fontsize=8)
            fig.tight_layout()

            path = out_dir / f"{tuned['experiment']}_{alg}_{i:02d}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(path)
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tuned", help="path to tuned/<experiment>.json")
    p.add_argument("--out", default="figures/tuning")
    a = p.parse_args()
    for path in plot_history(a.tuned, a.out):
        print(path)


if __name__ == "__main__":
    main()
