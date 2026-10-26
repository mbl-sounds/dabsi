import json

import numpy as np

from dbsi.experiments import EXPERIMENTS
from dbsi.tuning import apply_tuned, tune


def test_tune_resume_apply(tmp_path):
    exp = dict(EXPERIMENTS["wgn_smoke"])
    exp["algs"] = {k: exp["algs"][k] for k in ("dabsi", "dgd", "mcqn")}
    spaces = dict(
        dabsi=dict(mu=[0.1, 0.4]),
        dgd=dict(alpha=[1e-7, 1e-6, 1e-3]),
        mcqn=dict(mu=[0.05, 0.2, 0.6]),
    )
    path = tmp_path / "tuned.json"
    kw = dict(runs=2, min_frames=20, max_frames=80, spaces=spaces, log=lambda *_: None)

    tuned = tune(exp, str(tmp_path / "e"), str(path), algs=["dabsi", "dgd"], **kw)
    assert set(tuned["params"]) == {"dabsi", "dgd"}
    assert tuned["params"]["dabsi"]["mu"] in (0.1, 0.4) and tuned["params"]["dgd"][
        "alpha"
    ] in (1e-7, 1e-6, 1e-3)

    # resume: tuned algorithms are kept, the missing one is added
    before = json.loads(path.read_text())["params"]
    tuned = tune(exp, str(tmp_path / "e"), str(path), **kw)
    assert set(tuned["params"]) == {"dabsi", "dgd", "mcqn"}
    assert {k: tuned["params"][k] for k in before} == before

    # same call, same result
    other = tmp_path / "other.json"
    again = tune(exp, str(tmp_path / "e2"), str(other), **kw)
    assert again["params"] == tuned["params"]

    out = apply_tuned(exp, str(path))
    assert (
        out["algs"]["mcqn"]["mu"] == tuned["params"]["mcqn"]["mu"]
        and out["algs"]["mcqn"]["eta"] == exp["algs"]["mcqn"]["eta"]
    )
    assert (
        exp["algs"]["mcqn"]["mu"] == EXPERIMENTS["wgn_smoke"]["algs"]["mcqn"]["mu"]
    )  # original untouched
    assert np.isfinite(list(tuned["history"]["dabsi"][0]["scores"].values())).any()
