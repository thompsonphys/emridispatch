"""Unit tests for the workbench utilities.

Nothing here needs lisatools, FEW, or a GPU: the parameter/measure plumbing is
tested against a stub model and the toy Gaussian model.
"""

import numpy as np
import pytest

from emridispatch.parameters import NDIM, PARAM_NAMES
from emridispatch.response.toy import ToyGaussianLikelihood

INJECTION = {
    "mass_1": 1.0e6,
    "mass_2": 10.0,
    "a": 0.0,
    "p": 10.0,
    "e": 0.1,
    "x": 1.0,
    "q_k": 1.0,
    "phi_k": 1.5707963267948966,
    "q_s": 1.0,
    "phi_s": 1.5707963267948966,
    "luminosity_distance": 1.0,
    "phi_phi": 1.5707963267948966,
    "phi_theta": 1.5707963267948966,
    "phi_r": 1.5707963267948966,
}


@pytest.fixture
def toy_cfg_text(tmp_path):
    inj = "\n".join(f"  {k}: {v!r}" for k, v in INJECTION.items())
    return f"""
injection:
{inj}

data:
  response: toy
  duration: 0.3
  delta_t: 10.0
  inj_snr: 30.0
  channels: [A, E]

sampler:
  nsamples: 10

prior:
  fisher: none
  angle_sigma: 0.05
  fisher_use_gpu: false

reparam:
  mode: off
  idx: [0, 1, 2, 3, 4, 5]

run:
  outdir: {str(tmp_path / "nonexistent_run")!r}
  seed: 1
"""


@pytest.fixture
def toy_model():
    return ToyGaussianLikelihood(INJECTION, sigma_scale=0.05, snr=30.0)


def test_truth_is_injection_vector(toy_model):
    from emridispatch.workbench import truth

    t = truth(toy_model)
    assert t.shape == (NDIM,)
    assert t[0] == pytest.approx(np.log(1.0e6))
    assert t[1] == pytest.approx(np.log(10.0))
    assert t[3] == pytest.approx(10.0)


def test_to_vector_from_dict_matches_truth(toy_model):
    from emridispatch.workbench import to_vector, truth

    assert np.allclose(to_vector(INJECTION), truth(toy_model))


def test_to_vector_passes_through_arrays():
    from emridispatch.workbench import to_vector

    v = np.arange(NDIM, dtype=float)
    assert np.allclose(to_vector(v), v)


def test_to_vector_rejects_wrong_length():
    from emridispatch.workbench import to_vector

    with pytest.raises(ValueError, match="12"):
        to_vector(np.zeros(5))


def test_to_physical_round_trips(toy_model):
    from emridispatch.workbench import to_physical, to_vector, truth

    v = truth(toy_model)
    assert np.allclose(to_vector(to_physical(toy_model, v)), v)


def test_to_physical_fills_unsampled_from_injection(toy_model):
    from emridispatch.workbench import to_physical, truth

    phys = to_physical(toy_model, truth(toy_model))
    assert phys["x"] == pytest.approx(INJECTION["x"])
    assert phys["phi_theta"] == pytest.approx(INJECTION["phi_theta"])


def test_offset_is_additive_in_sampling_coords(toy_model):
    from emridispatch.workbench import offset, truth

    t = truth(toy_model)
    v = offset(toy_model, p=+0.05, ln_m1=-0.01)
    assert v[PARAM_NAMES.index("p")] == pytest.approx(t[3] + 0.05)
    assert v[PARAM_NAMES.index("ln_m1")] == pytest.approx(t[0] - 0.01)
    assert np.allclose(np.delete(v, [0, 3]), np.delete(t, [0, 3]))


def test_offset_rejects_unknown_name(toy_model):
    from emridispatch.workbench import offset

    with pytest.raises(ValueError, match="mass_1"):
        offset(toy_model, mass_1=1.0)


def test_load_returns_cfg_and_model(tmp_path, toy_cfg_text):
    from emridispatch.workbench import load

    path = tmp_path / "cfg.yaml"
    path.write_text(toy_cfg_text)

    cfg, model = load(str(path))

    assert cfg.data.response == "toy"
    assert isinstance(model, ToyGaussianLikelihood)
    assert model.injection_parameters["mass_2"] == pytest.approx(10.0)


def test_load_writes_nothing(tmp_path, toy_cfg_text):
    from emridispatch.workbench import load

    path = tmp_path / "cfg.yaml"
    path.write_text(toy_cfg_text)
    before = set(p.name for p in tmp_path.iterdir())

    load(str(path))

    assert set(p.name for p in tmp_path.iterdir()) == before


def test_lnlike_peaks_at_truth(toy_model):
    from emridispatch.workbench import lnlike, offset, truth

    assert lnlike(toy_model, truth(toy_model)) == pytest.approx(0.0)
    assert lnlike(toy_model, offset(toy_model, p=0.5)) < 0.0


def test_lnlike_accepts_dict_and_vector(toy_model):
    from emridispatch.workbench import lnlike, truth

    assert lnlike(toy_model, INJECTION) == pytest.approx(
        lnlike(toy_model, truth(toy_model)))


def test_lnlike_matches_model_call(toy_model):
    from emridispatch.workbench import lnlike, offset

    v = offset(toy_model, e=0.01)
    assert lnlike(toy_model, v) == pytest.approx(toy_model(v))


def test_signal_functions_reject_toy_model(toy_model):
    from emridispatch import workbench as wbk

    with pytest.raises(TypeError, match="no signal generator"):
        wbk.signal(toy_model, INJECTION)


def test_lnlike_propagates_model_type_error():
    from emridispatch.workbench import lnlike

    class _Exploding:
        injection_parameters = dict(INJECTION)

        def evaluate_likelihood(self, payload, full=False):
            raise TypeError("boom from inside the model")

    with pytest.raises(TypeError, match="boom from inside the model"):
        lnlike(_Exploding(), INJECTION, full=True)


def test_lisatools_model_exposes_time_domain_seam():
    from emridispatch.response.lisatools import LisatoolsEMRILikelihood

    assert callable(getattr(LisatoolsEMRILikelihood, "generate_time_domain", None))


def test_snr_hand_computed():
    from emridispatch.workbench import snr
    from workbench_stub import StubModel, _Domain

    model = StubModel()
    ones = np.ones((2, 4), dtype=complex)
    model.sensitivity_matrix.sens_mat = np.ones((2, 4))
    model.sensitivity_matrix.invC = np.ones((2, 4))
    model.sensitivity_matrix.differential_component = 1.0
    model.data_residual_array = _Domain(ones, np.arange(4) * 1.0, 1.0)
    model.analysis_container._data = model.data_residual_array
    model.analysis_container._sens = model.sensitivity_matrix

    result = snr(model, _Domain(ones, np.arange(4) * 1.0, 1.0), per_channel=True)

    assert result.optimal == pytest.approx(np.sqrt(32.0))
    assert result.detected == pytest.approx(np.sqrt(32.0))
    for chan in result.per_channel.values():
        assert chan.optimal == pytest.approx(4.0)
        assert chan.detected == pytest.approx(4.0)


def test_snr_per_channel_sums_to_total():
    from emridispatch.workbench import signal, snr
    from workbench_stub import CHANNELS, StubModel

    model = StubModel()
    template = signal(model, {"e": 0.1})

    result = snr(model, template, per_channel=True)

    assert sorted(result.per_channel) == sorted(CHANNELS)
    per_chan_sq = sum(v.optimal ** 2 for v in result.per_channel.values())
    assert per_chan_sq == pytest.approx(result.optimal ** 2, rel=1e-10)
    per_chan_det = sum(v.detected * v.optimal
                       for v in result.per_channel.values())
    assert per_chan_det == pytest.approx(result.detected * result.optimal,
                                         rel=1e-10)


def test_snr_per_channel_omitted_by_default():
    from emridispatch.workbench import signal, snr
    from workbench_stub import StubModel

    model = StubModel()
    assert snr(model, signal(model, {"e": 0.1})).per_channel is None


def test_snr_template_reuse_generates_once():
    from emridispatch.workbench import lnlike, signal, snr
    from workbench_stub import StubModel

    model = StubModel()
    template = signal(model, {"e": 0.1})
    before = model.calls

    snr(model, template)
    lnlike(model, template)

    assert model.calls == before


def test_snr_rejects_phase_maximize_with_per_channel():
    from emridispatch.workbench import signal, snr
    from workbench_stub import StubModel

    model = StubModel()
    template = signal(model, {"e": 0.1})
    with pytest.raises(ValueError, match="global phase"):
        snr(model, template, phase_maximize=True, per_channel=True)


def test_overlap_of_data_with_itself_is_one():
    from emridispatch.workbench import overlap
    from workbench_stub import StubModel

    model = StubModel()
    assert overlap(model, model.data_residual_array) == pytest.approx(1.0)


def test_overlap_is_bounded():
    from emridispatch.workbench import overlap, signal
    from workbench_stub import StubModel

    model = StubModel()
    value = overlap(model, signal(model, {"e": 0.1}))
    assert -1.0 <= value <= 1.0


def test_measure_generates_waveform_once():
    from emridispatch.workbench import measure
    from workbench_stub import StubModel

    model = StubModel()
    before = model.calls

    result = measure(model, {"e": 0.2})

    assert model.calls == before + 1
    assert result.snr.optimal > 0.0
    assert result.lnlike_full == pytest.approx(result.lnlike + 1000.0)
    assert -1.0 <= result.overlap <= 1.0


def test_measure_per_channel_flows_through():
    from emridispatch.workbench import measure
    from workbench_stub import CHANNELS, StubModel

    result = measure(StubModel(), {"e": 0.2}, per_channel=True)
    assert sorted(result.snr.per_channel) == sorted(CHANNELS)


def test_injection_template_is_noiseless_and_cached():
    from emridispatch.workbench import injection_template
    from workbench_stub import StubModel

    model = StubModel()
    first = injection_template(model)
    calls_after_first = model.calls
    second = injection_template(model)

    assert second is first
    assert model.calls == calls_after_first


def test_noise_is_data_minus_injection():
    from emridispatch.workbench import injection_template, noise
    from workbench_stub import StubModel

    model = StubModel()
    expected = (np.asarray(model.data_residual_array.data_res_arr.arr)
                - np.asarray(injection_template(model).data_res_arr.arr))

    assert np.allclose(noise(model), expected)


def test_noise_shape_matches_data():
    from emridispatch.workbench import noise
    from workbench_stub import NCHAN, NF, StubModel

    assert noise(StubModel()).shape == (NCHAN, NF)


def test_noise_warns_when_add_noise_off(caplog):
    from emridispatch.workbench import noise
    from workbench_stub import StubModel

    model = StubModel()
    model.add_noise = False
    with caplog.at_level("WARNING"):
        noise(model)

    assert "add_noise" in caplog.text


def test_noise_silent_when_add_noise_on(caplog):
    from emridispatch.workbench import noise
    from workbench_stub import StubModel

    model = StubModel()
    model.add_noise = True
    with caplog.at_level("WARNING"):
        noise(model)

    assert "add_noise" not in caplog.text


def _write_cfg(tmp_path, text):
    from emridispatch.config import load_config

    path = tmp_path / "cfg.yaml"
    path.write_text(text)
    return load_config(str(path))


def test_prior_from_config_builds_from_heuristic_fisher(tmp_path, toy_cfg_text):
    from emridispatch.priors import JointPrior
    from emridispatch.workbench import prior_from_config

    cfg = _write_cfg(tmp_path, toy_cfg_text)
    prior = prior_from_config(cfg)

    assert isinstance(prior, JointPrior)
    assert prior.ndim == NDIM
    assert prior.names == PARAM_NAMES


def test_prior_from_config_writes_nothing(tmp_path, toy_cfg_text):
    from emridispatch.workbench import prior_from_config

    cfg = _write_cfg(tmp_path, toy_cfg_text)
    outdir = tmp_path / "run_out"
    outdir.mkdir()
    cfg.run.outdir = str(outdir)

    prior_from_config(cfg)

    assert list(outdir.iterdir()) == []


def test_prior_from_config_prefers_prior_spec_json(tmp_path, toy_cfg_text):
    import json

    from emridispatch.workbench import prior_from_config

    cfg = _write_cfg(tmp_path, toy_cfg_text)
    outdir = tmp_path / "run_out"
    outdir.mkdir()
    cfg.run.outdir = str(outdir)
    spec = [{"name": n, "type": "uniform", "minimum": -3.0, "maximum": 7.0}
            for n in PARAM_NAMES]
    (outdir / "prior_spec.json").write_text(json.dumps(spec))

    prior = prior_from_config(cfg)

    assert np.allclose(prior.mins, -3.0)
    assert np.allclose(prior.maxes, 7.0)


def test_prior_from_config_gates_sef(tmp_path, toy_cfg_text):
    from emridispatch.workbench import prior_from_config

    cfg = _write_cfg(tmp_path, toy_cfg_text)
    cfg.prior.fisher = "sef"

    with pytest.raises(RuntimeError, match="allow_fisher"):
        prior_from_config(cfg)


def test_prior_from_config_uncalibrated_builds_no_model(tmp_path, toy_cfg_text,
                                                        monkeypatch):
    import emridispatch.response as response_mod
    from emridispatch.workbench import prior_from_config

    cfg = _write_cfg(tmp_path, toy_cfg_text)
    cfg.data.inj_snr = None

    def _fail(_cfg):
        raise AssertionError("prior_from_config built an injection model")

    monkeypatch.setattr(response_mod, "build_injection_model", _fail)
    prior = prior_from_config(cfg)

    assert prior.ndim == NDIM


def test_prior_from_config_reuses_supplied_model(tmp_path, toy_cfg_text,
                                                 monkeypatch):
    import emridispatch.response as response_mod
    from emridispatch.response.toy import ToyGaussianLikelihood
    from emridispatch.workbench import prior_from_config

    cfg = _write_cfg(tmp_path, toy_cfg_text)
    cfg.data.inj_snr = 30.0
    model = ToyGaussianLikelihood(INJECTION, sigma_scale=0.05, snr=30.0)

    def _fail(_cfg):
        raise AssertionError("prior_from_config ignored the supplied model")

    monkeypatch.setattr(response_mod, "build_injection_model", _fail)
    prior = prior_from_config(cfg, model=model)

    assert prior.ndim == NDIM
