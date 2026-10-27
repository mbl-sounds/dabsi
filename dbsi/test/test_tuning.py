import json

import numpy as np
import pytest

from dbsi.experiments import EXPERIMENTS, alg_spec, point_key
from dbsi.tuning import apply_tuned, tune


def _job(snr):
    return dict(run=0, M=3, N_i=2, SNR=snr, signal="white")


def test_tune_resume_apply(tmp_path):
    exp = dict(EXPERIMENTS["wgn_smoke"])
    exp["algs"] = {k: exp["algs"][k] for k in ("dabsi", "dgd", "mcqn")}
    exp["grid"] = dict(M=[3], N_i=[2], SNR=[20, 30], signal=["white"])
    spaces = dict(
        dabsi=dict(mu=[0.1, 0.4]),
        dgd=dict(alpha=[1e-7, 1e-6, 1e-3]),
        mcqn=dict(mu=[0.05, 0.2, 0.6]),
    )
    path = tmp_path / "tuned.json"
    kw = dict(runs=2, min_frames=20, max_frames=80, spaces=spaces, log=lambda *_: None)
    lo, hi = point_key(_job(20)), point_key(_job(30))

    tuned = tune(exp, str(tmp_path / "e"), str(path), algs=["dabsi", "dgd"], **kw)
    assert set(tuned["params"]) == {lo, hi}
    for pk in (lo, hi):
        assert set(tuned["params"][pk]) == {"dabsi", "dgd"}
        assert tuned["params"][pk]["dabsi"]["mu"] in (0.1, 0.4)
        assert tuned["params"][pk]["dgd"]["alpha"] in (1e-7, 1e-6, 1e-3)

    # resume: tuned algorithms are kept, the missing one is added
    before = json.loads(path.read_text())["params"]
    tuned = tune(exp, str(tmp_path / "e"), str(path), **kw)
    for pk in (lo, hi):
        assert set(tuned["params"][pk]) == {"dabsi", "dgd", "mcqn"}
        assert all(tuned["params"][pk][k] == v for k, v in before[pk].items())

    # same call, same result
    other = tmp_path / "other.json"
    again = tune(exp, str(tmp_path / "e2"), str(other), **kw)
    assert again["params"] == tuned["params"]

    out = apply_tuned(exp, str(path))
    assert alg_spec(out, _job(20), "mcqn")["mu"] == tuned["params"][lo]["mcqn"]["mu"]
    assert alg_spec(out, _job(30), "mcqn")["mu"] == tuned["params"][hi]["mcqn"]["mu"]
    assert alg_spec(out, _job(20), "mcqn")["eta"] == exp["algs"]["mcqn"]["eta"]
    assert (
        exp["algs"]["mcqn"]["mu"] == EXPERIMENTS["wgn_smoke"]["algs"]["mcqn"]["mu"]
    )  # original untouched
    with pytest.raises(KeyError):  # a grid point that was not tuned
        alg_spec(out, _job(40), "mcqn")
    assert np.isfinite(list(tuned["history"][lo]["dabsi"][0]["scores"].values())).any()
