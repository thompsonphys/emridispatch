import numpy as np
import pytest

from conftest import STUB_TABLE
from emridispatch.noise import (
    NOISE_MODEL, channel_noise_psd, noise_sens_kwargs, per_channel_noise_kwargs,
    sensitivity_spec)


def test_spec_tdi_off_is_two_direct_channels():
    sens, names = sensitivity_spec("off", None, STUB_TABLE)
    assert names == ["I", "II"]
    assert sens == ["LISASens", "LISASens"]


def test_spec_tdi_off_ignores_requested_channels():
    _, names = sensitivity_spec("off", ["A", "E"], STUB_TABLE)
    assert names == ["I", "II"]


def test_spec_default_channels():
    sens, names = sensitivity_spec("2nd generation", None, STUB_TABLE)
    assert names == ["A", "E"]
    assert sens == ["A2", "E2"]


def test_spec_explicit_channels():
    sens, names = sensitivity_spec("1st generation", ["A", "E", "T"], STUB_TABLE)
    assert names == ["A", "E", "T"]
    assert sens == ["A1", "E1", "T1"]


def test_spec_unknown_generation_lists_choices():
    with pytest.raises(ValueError, match="1st generation.*2nd generation"):
        sensitivity_spec("3rd generation", None, STUB_TABLE)


def test_spec_unknown_channel():
    with pytest.raises(ValueError, match="unknown TDI channel"):
        sensitivity_spec("2nd generation", ["A", "Q"], STUB_TABLE)


def test_noise_kwargs_without_foreground():
    assert noise_sens_kwargs(0.3, False) == {"model": NOISE_MODEL}


def test_noise_kwargs_with_foreground():
    pytest.importorskip("lisatools")
    from lisatools.utils.constants import YRSID_SI

    kwargs = noise_sens_kwargs(0.3, True)
    assert kwargs["model"] == NOISE_MODEL
    assert kwargs["stochastic_params"] == (0.3 * YRSID_SI,)


class _StubSens:
    def __init__(self, scale):
        self.scale = scale

    def get_Sn(self, f, **kwargs):
        self.seen = kwargs
        return self.scale * np.asarray(f, float)


def test_channel_noise_psd_dispatches_to_its_class():
    a, t = _StubSens(1.0), _StubSens(7.0)
    f = np.array([1e-3, 2e-3])
    assert np.allclose(channel_noise_psd(f, sens_cls=a, model="scirdv1"), f)
    assert np.allclose(channel_noise_psd(f, sens_cls=t, model="scirdv1"), 7.0 * f)
    assert a.seen == {"model": "scirdv1"}


def test_per_channel_noise_kwargs_one_dict_per_channel():
    kwargs = per_channel_noise_kwargs(0.3, False, ["A2", "E2", "T2"])
    assert [kw["sens_cls"] for kw in kwargs] == ["A2", "E2", "T2"]
    assert all(kw["model"] == NOISE_MODEL for kw in kwargs)
    assert all("stochastic_params" not in kw for kw in kwargs)


def test_per_channel_noise_kwargs_shares_foreground():
    pytest.importorskip("lisatools")
    kwargs = per_channel_noise_kwargs(0.3, True, ["A2", "T2"])
    assert len({kw["stochastic_params"] for kw in kwargs}) == 1
    assert kwargs[0]["stochastic_params"] == noise_sens_kwargs(
        0.3, True)["stochastic_params"]


def test_per_channel_noise_kwargs_does_not_mutate_shared_dict():
    kwargs = per_channel_noise_kwargs(0.3, False, ["A2", "E2"])
    kwargs[0]["model"] = "mangled"
    assert kwargs[1]["model"] == NOISE_MODEL


def test_aet_psds_are_distinct():
    pytest.importorskip("lisatools")

    sens, names = sensitivity_spec("2nd generation", ["A", "E", "T"])
    assert names == ["A", "E", "T"]
    f = np.geomspace(1e-4, 1e-1, 64)
    psds = [channel_noise_psd(f, **kw)
            for kw in per_channel_noise_kwargs(0.3, False, sens)]
    assert np.allclose(psds[0], psds[1], rtol=0.0, atol=0.0)
    assert not np.allclose(psds[0], psds[2], rtol=1e-3, atol=0.0)


def test_load_sensitivity_table_covers_generations():
    pytest.importorskip("lisatools")
    from emridispatch.noise import load_sensitivity_table

    table = load_sensitivity_table()
    assert set(table) == {"off", "1st generation", "2nd generation"}
    assert table["off"]["I"] is table["off"]["II"]
    assert set(table["2nd generation"]) == {"A", "E", "T"}
