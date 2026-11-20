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


def test_misalignment_ignores_complex_scale():
    h_true = np.array([[1.0 + 2.0j, 3.0 - 1.0j], [2.0 - 1.0j, 1.0 + 3.0j]])
    h_est = 1j * h_true

    got = misalignment(torch.as_tensor(h_est), torch.as_tensor(h_true)).numpy()
    np.testing.assert_allclose(got, 0.0, atol=1e-14)
    np.testing.assert_allclose(
        su.NPM(h_est.reshape(-1, 1), h_true.reshape(-1, 1)), 0.0, atol=1e-14
    )


def test_misalignment_distinguishes_relative_channel_phase():
    h_true = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=complex)
    h_est = np.array([[1.0, 0.0], [1.0j, 0.0]], dtype=complex)

    got = misalignment(torch.as_tensor(h_est), torch.as_tensor(h_true)).numpy()
    np.testing.assert_allclose(got, [1 / np.sqrt(2), 0.0, 0.0], atol=1e-14)


def test_misalignment_matches_complex_least_squares():
    h_true = np.array([[1.0 + 2.0j, 3.0 - 1.0j], [2.0 - 1.0j, 1.0 + 3.0j]])
    h_est = np.array([[2.0 - 1.0j, 1.0 + 1.0j], [1.0 + 2.0j, 2.0 + 1.0j]])

    def expected(a, b):
        scale = np.vdot(a, b) / np.vdot(a, a)
        return np.linalg.norm(b - scale * a) / np.linalg.norm(b)

    per = np.array([expected(a, b) for a, b in zip(h_est, h_true)])
    reference = [expected(h_est.ravel(), h_true.ravel()), per.mean(), per.min()]
    got = misalignment(torch.as_tensor(h_est), torch.as_tensor(h_true)).numpy()
    np.testing.assert_allclose(got, reference, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        su.getMisalignment(
            h_est.reshape(-1, 1), h_true.reshape(-1, 1), ["NPM", "NPMA", "NPMM"], 2
        ),
        reference,
        rtol=1e-12,
        atol=1e-14,
    )


def test_cython_misalignment_ignores_complex_scale():
    su_cy = pytest.importorskip("lib.simulation_utils_cy")
    h_true = np.array([[1.0 + 2.0j], [3.0 - 1.0j]])
    np.testing.assert_allclose(su_cy.NPM(1j * h_true, h_true), 0.0, atol=1e-14)


def test_misalignment_of_real_signals_matches_their_dfts():
    h_true = np.array([[1.0, 2.0, -1.0, 3.0], [2.0, -1.0, 1.0, 1.0]])
    h_est = np.array([[2.0, -1.0, 1.0, 3.0], [1.0, 2.0, -1.0, 2.0]])
    time = misalignment(torch.as_tensor(h_est), torch.as_tensor(h_true)).numpy()
    frequency = misalignment(
        torch.as_tensor(np.fft.fft(h_est, axis=-1)),
        torch.as_tensor(np.fft.fft(h_true, axis=-1)),
    ).numpy()
    np.testing.assert_allclose(frequency, time, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("SNR", [40, 20])
def test_wgn_old_vs_new(SNR):
    rng = np.random.default_rng(3)
    L, M, T = 16, 4, 30
    scn = generate_wgn_scenario(rng, M, SNR, L=L, nr_samples=(T + 3) * L, partitions=10)
    params = dict(rho=1.0, mu=0.4, eta=0.98, lambd=0.0, gev_reg=1e-9, gev_lim=4.0)
    cfg = DABSIConfig(L=L, eta=0.98, rho=1.0, mu=0.4, lambd=0.0, da=True)

    ref = run_reference(scn, params, np.eye(L), max_frames=T)
    new = run_dabsi(cfg, [scn], sigma_inv=np.eye(L), max_frames=T, vad_rule="center")[0]
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
