"""Harness tests: misalignment metric vs legacy implementation, and old-vs-new on a small WGN scenario."""

import numpy as np
import pytest
import torch

from dbsi.config import DABSIConfig
from dbsi.harness import generate_wgn_scenario, misalignment, run_dabsi, run_reference

su = pytest.importorskip("lib.simulation_utils")
pytest.importorskip("lib.admm")


def test_misalignment_matches_legacy():
    rng = np.random.default_rng(1)
    M, L = 3, 16
    h_true = rng.normal(size=(M, L)) + 1j * rng.normal(size=(M, L))
    h_est = h_true * (0.7 - 0.2j) + 0.3 * (
        rng.normal(size=(M, L)) + 1j * rng.normal(size=(M, L))
    )
    ref = su.getMisalignment(
        h_est.reshape(-1, 1), h_true.reshape(-1, 1), ["NPM", "NPMA", "NPMM"], L
    )
    got = misalignment(torch.as_tensor(h_est), torch.as_tensor(h_true)).numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-12)


@pytest.mark.parametrize("SNR", [40, 20])
def test_wgn_old_vs_new(SNR):
    rng = np.random.default_rng(3)
    L, M, N_i, T = 16, 4, 2, 30
    scn = generate_wgn_scenario(
        rng, M, N_i, SNR, L=L, nr_samples=(T + 3) * L, partitions=10
    )
    params = dict(rho=1.0, mu=0.4, eta=0.98, lambd=0.0, gev_reg=1e-9, gev_lim=4.0)
    cfg = DABSIConfig(L=L, eta=0.98, rho=1.0, mu=0.4, lambd=0.0, da=True)

    ref = run_reference(scn, params, np.eye(L), max_frames=T)
    new = run_dabsi(cfg, [scn], sigma_inv=np.eye(L), max_frames=T)[0]
    assert ref.shape == new.shape
    np.testing.assert_allclose(new, ref, rtol=1e-6, atol=1e-8)


def test_remove_initial_delay_drops_pre_direct_sound_samples():
    rir = np.zeros(200)
    rir[10] = 0.2
    rir[35] = 1.0
    rir[90] = 0.4
    aligned = su.removeInitialDelayFromRIRs([np.array([rir])], L=64, all=False)
    assert aligned.shape == (64, 1)
    assert np.argmax(np.abs(aligned[:, 0])) == 0
