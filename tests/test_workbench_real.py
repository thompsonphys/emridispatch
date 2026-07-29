"""Real-stack checks: the unit suite runs against a stub, which cannot catch a
mismatch with the actual lisatools container shape."""

import os

import pytest

pytest.importorskip("few")
pytest.importorskip("lisatools.response")

CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples", "eryn_config.yaml")


def test_injection_template_round_trips_on_real_stack():
    from emridispatch.workbench import (
        injection_template, load, measure, overlap, snr)

    _cfg, model = load(CONFIG)
    h = injection_template(model)

    assert snr(model, h).optimal == pytest.approx(model.optimal_snr, rel=1e-6)
    assert overlap(model, h) == pytest.approx(1.0, abs=1e-6)

    result = measure(model, h, per_channel=True)
    per_chan_sq = sum(v.optimal ** 2 for v in result.snr.per_channel.values())
    assert per_chan_sq == pytest.approx(result.snr.optimal ** 2, rel=1e-8)


def test_lnlike_peaks_at_the_truth():
    """__call__ end to end: the stub tests check the template dict, not that a
    waveform built from it reproduces the injection."""
    import numpy as np

    from emridispatch.parameters import NDIM, PARAM_NAMES, truth_vector
    from emridispatch.response.lisatools import LisatoolsEMRILikelihood

    snr = 30.0
    injection = dict(mass_1=1e6, mass_2=10.0, a=0.5, p=10.0, e=0.1, x=1.0,
                     luminosity_distance=1.0, q_s=1.0, phi_s=1.5, q_k=1.0,
                     phi_k=1.5, phi_phi=1.5, phi_theta=0.0, phi_r=1.5)
    model = LisatoolsEMRILikelihood(
        injection, duration=0.3, delta_t=10.0, injection_snr=snr,
        channel_list=["A", "E"], tdi="2nd generation", add_noise=False)

    truth = truth_vector(model.injection_parameters)
    # Noiseless: <d|h> - <h|h>/2 at the truth is exactly SNR^2/2.
    assert float(model(truth)) == pytest.approx(0.5 * snr ** 2, rel=1e-6)
    step = np.zeros(NDIM)
    step[PARAM_NAMES.index("p")] = 1e-3
    assert float(model(truth + step)) < 0.5 * snr ** 2
