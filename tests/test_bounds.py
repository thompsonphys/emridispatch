import numpy as np
import pytest

from emridispatch.bounds import (
    build_prior_bounds, fisher_cache_key, load_prior_bounds, save_prior_bounds)
from emridispatch.parameters import INTRINSIC_ORDER, NDIM
from emridispatch.reparam import Reparam

INJ = {
    "mass_1": 1e6, "mass_2": 10.0, "a": 0.0, "p": 10.0, "e": 0.1,
    "luminosity_distance": 1.0,
}
PREC = {"mass_1": 2.0, "mass_2": 1e-4, "a": 1e-5, "p": 1e-5, "e": 1e-5,
        "luminosity_distance": 0.05}
TRUTH = np.array([np.log(1e6), np.log(10.0), 0.0, 10.0, 0.1, 1.0,
                  1.0, 1.5, 1.0, 1.5, 1.5, 1.5])
IDX = np.arange(6)


def _build(box_scale=3.0, mode="auto"):
    cov = np.diag([PREC[p] ** 2 for p in INTRINSIC_ORDER])
    return build_prior_bounds(PREC, cov, list(INTRINSIC_ORDER), INJ, TRUTH,
                              mode, IDX, angle_sigma=0.05, ndim=NDIM,
                              box_scale=box_scale)


def test_build_box_geometry():
    mins, maxes, sample_cov, reparam = _build()
    assert mins.shape == (NDIM,) and maxes.shape == (NDIM,)
    assert np.all(maxes > mins)
    # Truth inside the box.
    assert np.all(TRUTH > mins) and np.all(TRUTH < maxes)
    # Mass rows are in log: exp bounds bracket the linear truth +/- 3 sigma.
    assert np.isclose(np.exp(mins[0]), 1e6 - 3 * PREC["mass_1"])
    assert np.isclose(np.exp(maxes[0]), 1e6 + 3 * PREC["mass_1"])
    # Angle rows carry fixed physical ranges.
    assert np.isclose(maxes[6], np.pi) and np.isclose(maxes[7], 2 * np.pi)
    # Proposal cov: intrinsic block log-mass Jacobian applied.
    assert np.isclose(sample_cov[0, 0], (PREC["mass_1"] / 1e6) ** 2)
    assert np.isclose(sample_cov[6, 6], 0.05 ** 2)


def test_save_load_roundtrip(tmp_path):
    mins, maxes, sample_cov, reparam = _build()
    path = str(tmp_path / "prior_bounds.npz")
    save_prior_bounds(path, mins, maxes, sample_cov, reparam,
                      box_scale=3.0, prec_dict=PREC, injection_parameters=INJ,
                      reparam_mode="auto")
    m2, x2, cov2, rp2, mode2 = load_prior_bounds(path, NDIM, IDX, "auto")
    assert np.allclose(m2, mins) and np.allclose(x2, maxes)
    assert np.allclose(cov2, sample_cov)
    assert np.allclose(rp2.R, reparam.R)
    assert mode2 == "auto"


def test_load_rescales_box(tmp_path):
    mins, maxes, sample_cov, reparam = _build(box_scale=3.0)
    path = str(tmp_path / "prior_bounds.npz")
    save_prior_bounds(path, mins, maxes, sample_cov, reparam,
                      box_scale=3.0, prec_dict=PREC, injection_parameters=INJ,
                      reparam_mode="auto")
    m5, x5, _, _, _ = load_prior_bounds(path, NDIM, IDX, "auto", box_scale=5.0)
    ref_m5, ref_x5, _, _ = _build(box_scale=5.0)
    assert np.allclose(m5, ref_m5) and np.allclose(x5, ref_x5)


def test_load_mode_mismatch_raises(tmp_path):
    mins, maxes, sample_cov, reparam = _build(mode="auto")
    path = str(tmp_path / "prior_bounds.npz")
    save_prior_bounds(path, mins, maxes, sample_cov, reparam,
                      box_scale=3.0, prec_dict=PREC, injection_parameters=INJ,
                      reparam_mode="auto")
    with pytest.raises(ValueError, match="reparam mode"):
        load_prior_bounds(path, NDIM, IDX, "grid")


def _save(path, fisher_key=None):
    mins, maxes, sample_cov, reparam = _build()
    save_prior_bounds(path, mins, maxes, sample_cov, reparam,
                      box_scale=3.0, prec_dict=PREC, injection_parameters=INJ,
                      reparam_mode="auto", fisher_key=fisher_key)


def test_fisher_key_null_channels_matches_default():
    assert (fisher_cache_key("2nd generation", True, 1.0, 10.0, None)
            == fisher_cache_key("2nd generation", True, 1.0, 10.0, ["A", "E"]))


def test_fisher_key_varies_with_each_ingredient():
    base = fisher_cache_key("2nd generation", True, 1.0, 10.0, ["A", "E"])
    others = [
        fisher_cache_key("1st generation", True, 1.0, 10.0, ["A", "E"]),
        fisher_cache_key("2nd generation", False, 1.0, 10.0, ["A", "E"]),
        fisher_cache_key("2nd generation", True, 2.0, 10.0, ["A", "E"]),
        fisher_cache_key("2nd generation", True, 1.0, 5.0, ["A", "E"]),
        fisher_cache_key("2nd generation", True, 1.0, 10.0, ["A", "E", "T"]),
    ]
    assert len(set(others) | {base}) == len(others) + 1


def test_fisher_key_ignores_channels_when_tdi_off():
    assert (fisher_cache_key("off", True, 1.0, 10.0, None)
            == fisher_cache_key("off", True, 1.0, 10.0, ["A", "E", "T"]))


def test_fisher_key_roundtrip(tmp_path):
    path = str(tmp_path / "prior_bounds.npz")
    key = fisher_cache_key("off", False, 1.0, 10.0, None)
    _save(path, fisher_key=key)
    mins, maxes, _, _, mode = load_prior_bounds(
        path, NDIM, IDX, "auto", fisher_key=key)
    assert mode == "auto" and mins.shape == (NDIM,) and maxes.shape == (NDIM,)


def test_fisher_key_mismatch_raises(tmp_path):
    path = str(tmp_path / "prior_bounds.npz")
    _save(path, fisher_key=fisher_cache_key("off", False, 1.0, 10.0, None))
    with pytest.raises(ValueError, match="Fisher-relevant config"):
        load_prior_bounds(path, NDIM, IDX, "auto",
                          fisher_key=fisher_cache_key(
                              "2nd generation", True, 1.0, 10.0, None))


def test_legacy_cache_without_fisher_key_warns(tmp_path, caplog):
    path = str(tmp_path / "prior_bounds.npz")
    _save(path)
    with caplog.at_level("WARNING"):
        mins, _, _, _, _ = load_prior_bounds(
            path, NDIM, IDX, "auto",
            fisher_key=fisher_cache_key("off", False, 1.0, 10.0, None))
    assert mins.shape == (NDIM,)
    assert any("fisher_key" in r.getMessage() for r in caplog.records)


def test_mode_off_identity(tmp_path):
    mins, maxes, sample_cov, reparam = _build(mode="off")
    assert isinstance(reparam, Reparam)
    x = TRUTH.copy()
    assert np.allclose(reparam.to_u(x), x)


def _manual_cfg(**prior_kw):
    from types import SimpleNamespace

    return SimpleNamespace(prior=SimpleNamespace(fisher="manual", **prior_kw))


def test_manual_sigmas_box_provider():
    from emridispatch.fisher import get_fisher_provider

    provider = get_fisher_provider(_manual_cfg(sigmas=dict(PREC)))
    assert provider.name == "manual"
    fr = provider.compute(INJ, duration=1.0, delta_t=10.0)
    assert fr.sigmas == {p: float(PREC[p]) for p in INTRINSIC_ORDER}
    assert np.allclose(fr.cov, np.diag([PREC[p] ** 2 for p in INTRINSIC_ORDER]))
    # The box built from sigmas is a truth +/- n*sigma rectangle.
    n = 25.0
    mins, maxes, sample_cov, _ = build_prior_bounds(
        fr.sigmas, fr.cov, fr.order, INJ, TRUTH, "auto", IDX,
        angle_sigma=0.05, ndim=NDIM, box_scale=n)
    assert np.isclose(np.exp(mins[0]), 1e6 - n * PREC["mass_1"])
    assert np.isclose(np.exp(maxes[0]), 1e6 + n * PREC["mass_1"])
    assert np.isclose(maxes[4], 0.1 + n * PREC["e"])
    # Angle rows keep full physical ranges, independent of the scale.
    assert np.isclose(maxes[6], np.pi) and np.isclose(maxes[7], 2 * np.pi)
    # Proposal covariance diag(sigma^2) (log-mass Jacobian on the mass rows).
    assert np.isclose(sample_cov[2, 2], PREC["a"] ** 2)
    assert np.isclose(sample_cov[0, 0], (PREC["mass_1"] / 1e6) ** 2)


def test_manual_provider_config_errors():
    from emridispatch.fisher import get_fisher_provider

    with pytest.raises(ValueError, match="prior.sigmas"):
        get_fisher_provider(_manual_cfg())
    incomplete = {k: v for k, v in PREC.items() if k != "e"}
    with pytest.raises(ValueError, match="missing"):
        get_fisher_provider(_manual_cfg(sigmas=incomplete))
