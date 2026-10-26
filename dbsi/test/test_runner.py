import numpy as np
import pytest

from dbsi import analysis as an
from dbsi.experiments import EXPERIMENTS, job_id, make_jobs, run_job
from dbsi.runner import git_info, run_experiment


def test_run_resume_crash_analysis(tmp_path):
    exp = dict(EXPERIMENTS["wgn_smoke"])
    assert run_experiment(exp, str(tmp_path), processes=1) == 0
    out = tmp_path / exp["name"]
    files = sorted(out.glob("*/*.npz"))
    assert len(files) == len(make_jobs(exp)) * len(exp["algs"])
    assert not list(out.glob("*/*.tmp"))

    # resume: nothing to do, files untouched
    mtimes = [f.stat().st_mtime_ns for f in files]
    assert run_experiment(exp, str(tmp_path), processes=1) == 0
    assert [f.stat().st_mtime_ns for f in files] == mtimes

    # crash after some algorithms of a job: missing ones are recomputed, result identical to a clean run
    j = make_jobs(exp)[0]
    jdir = out / job_id(j)
    ref = {f.stem: np.load(f)["mis"] for f in jdir.glob("*.npz")}
    for n in ("dgd", "mcqn"):
        (jdir / f"{n}.npz").unlink()
    (jdir / "spmcqn.tmp").write_bytes(b"partial")
    assert run_experiment(exp, str(tmp_path), processes=1) == 0
    for n, v in ref.items():
        assert np.array_equal(np.load(jdir / f"{n}.npz")["mis"], v), n
    np.testing.assert_allclose(ref["dabsi"], ref["dabsi_ref"], rtol=1e-4, atol=1e-5)

    # analysis: cache is built, reused and extended
    df = an.load(str(out))
    assert (out / "_cache.parquet").exists() and df.height == len(files)
    assert an.load(str(out)).height == df.height
    c = an.curves(df, "npm", alg="dabsi")
    assert c.shape[0] == 2 and np.isfinite(c).all()
    s = an.summary(df, "npm")
    assert s.height == len(exp["algs"]) and (s["n_total"] == 2).all()
    assert (s["scale_spread_db"] >= 0).all()


def test_room_scenarios_and_prior(tmp_path):
    exp = dict(EXPERIMENTS["room_smoke"])
    assert run_experiment(exp, str(tmp_path), processes=1) == 0
    df = an.load(str(tmp_path / exp["name"]))
    assert set(df["signal"]) == {"white", "stationary"} and df.height == 4
    assert list((tmp_path / exp["name"] / "_priors").glob("prior_*.npy"))


def test_room_job_reproducible_regardless_of_global_rng(tmp_path):
    exp = dict(EXPERIMENTS["room_smoke"])
    job = make_jobs(exp)[0]
    res = []
    for d, draws in (("a", 1), ("b", 1000)):
        np.random.rand(draws)  # different global RNG history before the job
        run_job(exp, job, str(tmp_path / d), git_info())
        jdir = tmp_path / d / job_id(job)
        res.append({n: np.load(jdir / f"{n}.npz")["mis"] for n in exp["algs"]})
    for n in exp["algs"]:
        assert np.array_equal(res[0][n], res[1][n]), n


def test_speech_scenario():
    import os

    from dbsi.experiments import LIBRISPEECH_DATASET, LIBRISPEECH_PATH, make_scenario

    if not os.path.isdir(os.path.join(LIBRISPEECH_PATH, LIBRISPEECH_DATASET)):
        pytest.skip("LibriSpeech not available in the working directory")
    exp = dict(EXPERIMENTS["wgn_smoke"])
    exp["scenario"] = dict(exp["scenario"], nr_samples=20_000)
    scn = make_scenario(exp, dict(run=0, M=3, N_i=2, SNR=20, signal="speech"))
    assert scn.noisy.shape[1] == 3
    assert np.isfinite(scn.noisy).all()
