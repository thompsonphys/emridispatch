"""Functional cover for the PSD notch.

The detector runs on the sensitivity alone, so most of it is reachable
without a waveform. Only the false-positive regression needs the real stack.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.fft import next_fast_len

pytest.importorskip("lisatools.domains")
pytest.importorskip("lisatools.sensitivity")

from lisatools.domains import FDSettings, TDSettings  # noqa: E402
from lisatools.utils.constants import YRSID_SI  # noqa: E402

from emridispatch.noise import (  # noqa: E402
    MIN_FREQ, channel_noise_psd, noise_sens_kwargs, sensitivity_spec)
from emridispatch.response.lisatools import EMRIInjectionGenerator  # noqa: E402

DELTA_T = 10.0
C_OVER_4L = 0.0299792


def _fd_settings(duration):
    n = int(round(duration * YRSID_SI / DELTA_T))
    td = TDSettings(next_fast_len(n, True), DELTA_T, force_backend="cpu")
    stub = SimpleNamespace(_lt=SimpleNamespace(FDSettings=FDSettings))
    return td, EMRIInjectionGenerator._fd_settings(stub, td)


def _groups(mask):
    idx = np.flatnonzero(mask)
    return np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)


@pytest.mark.parametrize("duration", [0.3, 4.0])
def test_the_detector_finds_only_the_tdi_null(duration):
    """A second group means the block-median baseline is flagging the band
    edge again: the median of block 0 is not a local baseline where the
    sensitivity is a steep power law."""
    f = np.asarray(_fd_settings(duration)[1].f_arr)
    sens_classes, _ = sensitivity_spec("2nd generation", ["A", "E", "T"])
    kwargs = noise_sens_kwargs(duration, foreground=True)
    sens_mat = np.asarray(
        [channel_noise_psd(f, cls, **kwargs) for cls in sens_classes])

    mask = EMRIInjectionGenerator._psd_null_mask(
        SimpleNamespace(psd_notch=1e-5, psd_notch_depth=2.0), f, sens_mat)

    assert mask is not None
    groups = _groups(mask)
    assert len(groups) == 1
    assert f[groups[0][0]] < C_OVER_4L < f[groups[0][-1]]


def test_min_freq_floors_the_analysis_band():
    td, settings = _fd_settings(0.3)
    f = np.asarray(settings.f_arr)
    df = 1.0 / (td.N * td.dt)
    assert f[0] >= MIN_FREQ
    assert f[0] - MIN_FREQ < 2.0 * df


def test_a_low_frequency_source_does_not_trip_a_false_psd_null():
    """m1=1e7 radiates near the band edge, where the analytic S_T collapses
    but the numerical response does not. Construction raised PSDNullError on
    a notch that cost 0.038% of the SNR."""
    pytest.importorskip("few")
    pytest.importorskip("lisatools.response")
    from emridispatch.response.lisatools import LisatoolsEMRILikelihood

    injection = dict(mass_1=1e7, mass_2=10.0, a=0.0, p=16.0, e=0.1, x=1.0,
                     luminosity_distance=1.0, q_s=1.0, phi_s=np.pi / 2,
                     q_k=1.0, phi_k=np.pi / 2, phi_phi=np.pi / 2,
                     phi_theta=0.0, phi_r=np.pi / 2)
    model = LisatoolsEMRILikelihood(
        injection, duration=0.3, delta_t=DELTA_T, injection_snr=None,
        channel_list=["A", "E", "T"], tdi="2nd generation",
        add_noise=False, psd_notch_strict=True)

    assert len(_groups(model._psd_notch_mask)) == 1
    assert model._notch_drift < 1e-6
