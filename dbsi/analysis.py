"""Fast loading and summary of experiment results.

``load`` reads all per-(job, alg) files once into a polars frame (one row per file, curves as list
columns) and caches it as ``<exp_dir>/_cache.parquet``; later calls only read new files.
Misalignments are stored linear (as in the legacy code); ``curves(..., db=True)`` converts to dB.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

METRICS = {"npm": 0, "npma": 1, "npmm": 2}


def _read(f: Path) -> dict:
    z = np.load(f)
    meta = json.loads(str(z["meta"]))
    mis, snr = z["mis"], np.asarray(meta["mic_snr_db"])
    return dict(
        file=f"{f.parent.name}/{f.name}",
        alg=meta["alg"],
        **meta["job"],
        npm=mis[:, 0].tolist(),
        npma=mis[:, 1].tolist(),
        npmm=mis[:, 2].tolist(),
        scale=z["scale"].tolist(),  # [T, M] relative channel scale in dB
        sec=float(z["sec"]),
        sec_per_frame=meta["sec_per_frame"],
        mic_snr_min=float(snr.min()),
        mic_snr_mean=float(snr.mean()),
        mic_snr_max=float(snr.max()),
        degree_mean=float(np.mean(meta["degree_in"])),
        vad_mean=float(np.mean(meta["vad_fraction"])),
    )


def load(exp_dir: str, cache: bool = True) -> pl.DataFrame:
    d = Path(exp_dir)
    cache_f = d / "_cache.parquet"
    old = pl.read_parquet(cache_f) if cache and cache_f.exists() else None
    known = set(old["file"]) if old is not None else set()
    new = [
        _read(f)
        for f in sorted(d.glob("*/*.npz"))
        if f"{f.parent.name}/{f.name}" not in known
    ]
    if not new and old is not None:
        return old
    df = pl.concat(
        [x for x in (old, pl.DataFrame(new) if new else None) if x is not None],
        how="diagonal",
    )
    if cache:
        tmp = cache_f.with_suffix(".tmp")
        df.write_parquet(tmp)
        tmp.replace(cache_f)
    return df


def params(df: pl.DataFrame) -> list[str]:
    """Grid-point columns (job parameters other than the run index)."""
    skip = {
        "file",
        "alg",
        "run",
        "npm",
        "npma",
        "npmm",
        "scale",
        "sec",
        "sec_per_frame",
        "mic_snr_min",
        "mic_snr_mean",
        "mic_snr_max",
        "degree_mean",
        "vad_mean",
    }
    return [c for c in df.columns if c not in skip]


def select(df: pl.DataFrame, **where) -> pl.DataFrame:
    for k, v in where.items():
        df = df.filter(
            pl.col(k).is_in(v) if isinstance(v, (list, tuple)) else pl.col(k) == v
        )
    return df.sort("run")


def curves(df: pl.DataFrame, metric: str = "npm", db: bool = True, **where) -> np.ndarray:
    """[R, T] misalignment curves of the selected rows (or [R, T, M] for ``scale``)."""
    a = np.asarray(select(df, **where)[metric].to_list(), dtype=float)
    return 20 * np.log10(a) if db and metric in METRICS else a


def summary(
    df: pl.DataFrame,
    metric: str = "npm",
    tail: int = 50,
    conv_db: float = -10.0,
    fail_db: float = -5.0,
) -> pl.DataFrame:
    """
    One row per (grid point, alg): the final misalignment [dB] (mean over the last ``tail`` frames of a
    run) as median/mean/IQR/std/min over runs, the first frame below ``conv_db`` (median over runs that
    reach it), failure rate (final above ``fail_db`` or non-finite), valid/total runs, the median
    relative-scale spread [dB] (max-min over channels, last ``tail`` frames) and runtime per frame.
    """
    rows = []
    for key, g in df.group_by(params(df) + ["alg"], maintain_order=True):
        a = np.asarray(g[metric].to_list(), dtype=float)
        db = 20 * np.log10(a)
        fin = np.nanmean(
            np.where(np.isfinite(db[:, -tail:]), db[:, -tail:], np.nan), axis=1
        )
        ok = np.isfinite(fin)
        reach = [np.argmax(r <= conv_db) for r in db if (r <= conv_db).any()]
        sc = np.asarray(g["scale"].to_list(), dtype=float)[:, -tail:]  # [R, tail, M]
        spread = np.median(sc.max(-1) - sc.min(-1), axis=1)
        v = fin[ok]
        q1, q3 = np.percentile(v, [25, 75]) if ok.any() else (np.nan, np.nan)
        rows.append(
            dict(
                zip(params(df) + ["alg"], key),
                n_valid=int(ok.sum()),
                n_total=len(a),
                median=float(np.median(v)) if ok.any() else np.nan,
                mean=float(v.mean()) if ok.any() else np.nan,
                q1=float(q1),
                q3=float(q3),
                std=float(v.std()) if ok.any() else np.nan,
                min=float(v.min()) if ok.any() else np.nan,
                conv_frame=float(np.median(reach)) if reach else np.nan,
                fail_rate=float(1 - np.mean(ok & (fin <= fail_db))),
                scale_spread_db=float(np.median(spread)),
                ms_per_frame=1e3 * float(g["sec_per_frame"].median()),
            )
        )
    return pl.DataFrame(rows)
