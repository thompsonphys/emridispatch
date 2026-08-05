import logging
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import STUB_TABLE
from emridispatch.config import INJECTION_KEYS
from emridispatch.parameters import NDIM
from emridispatch.response.lisatools import (
    LisatoolsEMRILikelihood, _DirectEMRIWaveform, _build_waveform_and_sens)


class _FakeResponseWrapper:
    def __init__(self, gen, Tobs, dt, **kwargs):
        self.gen = gen
        self.Tobs = Tobs
        self.dt = dt
        self.kwargs = kwargs


class _FakeOrbits:
    pass


def _stub_lt():
    return SimpleNamespace(
        sens_table=STUB_TABLE,
        ResponseWrapper=_FakeResponseWrapper,
        EqualArmlengthOrbits=_FakeOrbits,
    )


def _stub_few():
    return SimpleNamespace(
        GenerateEMRIWaveform=lambda *a, **k: ("few_gen", a, k))


def test_tdi_on_2nd_generation():
    wf, channels, sens = _build_waveform_and_sens(
        _stub_lt(), _stub_few(), "2nd generation", ["A", "E"], 0.3, 10.0)
    assert isinstance(wf, _FakeResponseWrapper)
    assert wf.kwargs["tdi"] == "2nd generation"
    assert wf.kwargs["tdi_chan"] == "AE"
    assert wf.Tobs == 0.3 and wf.dt == 10.0
    assert channels == ["A", "E"]
    assert sens == ["A2", "E2"]


def test_tdi_on_waveform_comes_from_few_with_padded_output():
    wf, _, _ = _build_waveform_and_sens(
        _stub_lt(), _stub_few(), "2nd generation", ["A", "E"], 0.3, 10.0)
    tag, args, kwargs = wf.gen
    assert args == ("FastKerrEccentricEquatorialFlux",)
    assert kwargs == {"sum_kwargs": {"pad_output": True}}


def test_tdi_on_response_configuration_is_passed_in_full():
    """Pins what EMRITDIWaveform used to supply implicitly."""
    wf, _, _ = _build_waveform_and_sens(
        _stub_lt(), _stub_few(), "2nd generation", ["A", "E", "T"], 1.0, 5.0)
    assert wf.kwargs["index_lambda"] == 8
    assert wf.kwargs["index_beta"] == 7
    assert wf.kwargs["t0"] == 30000.0
    assert wf.kwargs["order"] == 25
    assert wf.kwargs["flip_hx"] is True
    assert wf.kwargs["remove_sky_coords"] is False
    assert wf.kwargs["is_ecliptic_latitude"] is False
    assert isinstance(wf.kwargs["orbits"], _FakeOrbits)
    assert wf.kwargs["tdi_chan"] == "AET"


def test_tdi_on_1st_generation_default_channels():
    wf, channels, sens = _build_waveform_and_sens(
        _stub_lt(), _stub_few(), "1st generation", None, 0.3, 10.0)
    assert channels == ["A", "E"]
    assert sens == ["A1", "E1"]
    assert wf.kwargs["tdi"] == "1st generation"


def test_tdi_on_unknown_channel_raises():
    with pytest.raises(ValueError, match="unknown TDI channel"):
        _build_waveform_and_sens(
            _stub_lt(), _stub_few(), "2nd generation", ["A", "Q"], 0.3, 10.0)


def test_tdi_unknown_generation_raises():
    with pytest.raises(ValueError, match="unknown TDI generation"):
        _build_waveform_and_sens(
            _stub_lt(), _stub_few(), "3rd generation", ["A", "E"], 0.3, 10.0)


def test_tdi_unknown_generation_lists_choices():
    with pytest.raises(ValueError, match="1st generation.*2nd generation"):
        _build_waveform_and_sens(
            _stub_lt(), _stub_few(), "3rd generation", None, 0.3, 10.0)


def test_tdi_off_direct_waveform():
    wf, channels, sens = _build_waveform_and_sens(
        _stub_lt(), _stub_few(), "off", ["A", "E"], 0.3, 10.0)
    assert isinstance(wf, _DirectEMRIWaveform)
    assert channels == ["I", "II"]
    assert sens == ["LISASens", "LISASens"]
    tag, args, kwargs = wf.gen
    assert args == ("FastKerrEccentricEquatorialFlux",)
    assert kwargs["sum_kwargs"] == {"pad_output": True}
    assert kwargs["return_list"] is False
    assert kwargs["frame"] == "detector"


def test_direct_waveform_call_returns_both_polarizations():
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
    assert len(out) == 2
    np.testing.assert_array_equal(out[0], h.real)
    np.testing.assert_array_equal(out[1], -h.imag)


# --- __call__ failure reporting -------------------------------------------

VEC = np.full(NDIM, 0.5)


def _model(evaluate=None, log=None):
    """A real instance without __init__, which needs the whole lisatools stack.

    lnlike_failures is absent until the first failure records it, and
    log_lnlike_failures is left to the class default unless `log` is given.
    """
    model = object.__new__(LisatoolsEMRILikelihood)
    model.injection_parameters = {name: 1.0 for name in INJECTION_KEYS}
    model.evaluate_likelihood = evaluate or (lambda template: 0.0)
    if log is not None:
        model.log_lnlike_failures = log
    return model


def _raises(exc):
    def evaluate(_template):
        raise exc
    return evaluate


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING]


def test_failures_are_silent_by_default(caplog):
    """A sampler run must not gain log output it did not have before; neither
    impulse nor eryn reports a likelihood failure either."""
    assert LisatoolsEMRILikelihood.log_lnlike_failures is False
    model = _model(_raises(ValueError("p below separatrix")))
    with caplog.at_level(logging.WARNING, logger="emridispatch"):
        for _ in range(5):
            assert model(VEC) == -np.inf
    assert _warnings(caplog) == []


def test_failures_are_counted_whether_or_not_they_are_logged():
    """The count is the sampler-agnostic half: no log lines, so a run that
    fails at every point is still distinguishable from a hard injection."""
    model = _model(_raises(ValueError("p below separatrix")))
    for _ in range(5):
        assert model(VEC) == -np.inf
    assert model.lnlike_failures == {"waveform: ValueError": 5}


def test_a_bad_sampling_vector_counts_separately_from_a_waveform_failure():
    """A short vector is a programming error, never a physical condition, so
    it must not be pooled with out-of-domain waveforms."""
    model = _model()
    assert model(VEC[:-1]) == -np.inf
    assert model.lnlike_failures == {"parameters: IndexError": 1}


def test_opt_in_logging_reports_each_failure_type_once(caplog):
    """Per-type, not per-call, so enabling it cannot saturate run.log: a
    TypeError arriving after the ValueErrors is the new information."""
    model = _model(_raises(ValueError("domain")), log=True)
    with caplog.at_level(logging.WARNING, logger="emridispatch"):
        model(VEC)
        model(VEC)
        model.evaluate_likelihood = _raises(TypeError("bad container"))
        model(VEC)
        model(VEC)
    assert len(_warnings(caplog)) == 2
    assert "domain" in _warnings(caplog)[0].getMessage()
    assert model.lnlike_failures == {"waveform: ValueError": 2,
                                     "waveform: TypeError": 2}


def test_the_first_logged_failure_carries_a_traceback(caplog):
    """The point of opting in is locating a bug, not counting it."""
    model = _model(_raises(TypeError("bad container")), log=True)
    with caplog.at_level(logging.WARNING, logger="emridispatch"):
        model(VEC)
    assert _warnings(caplog)[0].exc_info is not None


def test_a_successful_call_records_no_failure(caplog):
    model = _model(lambda template: -3.5)
    with caplog.at_level(logging.WARNING, logger="emridispatch"):
        assert model(VEC) == -3.5
    assert _warnings(caplog) == []
    assert getattr(model, "lnlike_failures", {}) == {}
