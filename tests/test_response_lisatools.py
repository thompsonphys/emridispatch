from types import SimpleNamespace

import numpy as np
import pytest

from emridispatch.response.lisatools import (
    _DirectEMRIWaveform, _build_waveform_and_sens)


class _FakeTDIWaveform:
    def __init__(self, response_kwargs=None, T=None, dt=None):
        self.response_kwargs = response_kwargs
        self.T = T
        self.dt = dt


def _stub_lt():
    return SimpleNamespace(
        sens_by_channel={
            "1st generation": {"A": "A1", "E": "E1", "T": "T1"},
            "2nd generation": {"A": "A2", "E": "E2", "T": "T2"},
        },
        LISASens="LISASens",
        GenerateEMRIWaveform=lambda *a, **k: ("few_gen", a, k),
        EMRITDIWaveform=_FakeTDIWaveform,
    )


def test_tdi_on_2nd_generation():
    wf, channels, sens = _build_waveform_and_sens(
        _stub_lt(), "2nd generation", ["A", "E"], 0.3, 10.0)
    assert isinstance(wf, _FakeTDIWaveform)
    assert wf.response_kwargs["tdi"] == "2nd generation"
    assert wf.response_kwargs["tdi_chan"] == "AE"
    assert wf.T == 0.3 and wf.dt == 10.0
    assert channels == ["A", "E"]
    assert sens == ["A2", "E2"]


def test_tdi_on_1st_generation_default_channels():
    wf, channels, sens = _build_waveform_and_sens(
        _stub_lt(), "1st generation", None, 0.3, 10.0)
    assert channels == ["A", "E"]
    assert sens == ["A1", "E1"]
    assert wf.response_kwargs["tdi"] == "1st generation"


def test_tdi_on_unknown_channel_raises():
    with pytest.raises(ValueError, match="unknown TDI channel"):
        _build_waveform_and_sens(
            _stub_lt(), "2nd generation", ["A", "Q"], 0.3, 10.0)


def test_tdi_unknown_generation_raises():
    with pytest.raises(ValueError, match="unknown TDI generation"):
        _build_waveform_and_sens(
            _stub_lt(), "3rd generation", ["A", "E"], 0.3, 10.0)


def test_tdi_unknown_generation_lists_choices():
    with pytest.raises(ValueError, match="1st generation.*2nd generation"):
        _build_waveform_and_sens(
            _stub_lt(), "3rd generation", None, 0.3, 10.0)


def test_tdi_off_direct_waveform():
    wf, channels, sens = _build_waveform_and_sens(
        _stub_lt(), "off", ["A", "E"], 0.3, 10.0)
    assert isinstance(wf, _DirectEMRIWaveform)
    assert channels == ["I"]
    assert sens == ["LISASens"]
    tag, args, kwargs = wf.gen
    assert args == ("FastKerrEccentricEquatorialFlux",)
    assert kwargs["sum_kwargs"] == {"pad_output": True}
    assert kwargs["return_list"] is False


def test_direct_waveform_call_returns_single_real_channel():
    h = np.array([1 + 2j, 3 - 4j])
    calls = {}

    def gen(*params, T=None, dt=None):
        calls["params"] = params
        calls["T"] = T
        calls["dt"] = dt
        return h

    wf = _DirectEMRIWaveform(gen, 0.3, 10.0)
    out = wf(1.0, 2.0, 3.0)
    assert calls["params"] == (1.0, 2.0, 3.0)
    assert calls["T"] == 0.3 and calls["dt"] == 10.0
    assert len(out) == 1
    np.testing.assert_array_equal(out[0], h.real)
