"""Everything here is derived from INJECTION_KEYS / PARAM_NAMES, so a new
injection parameter is covered without editing this file whether or not the
sampling vector grows a row for it."""

from types import SimpleNamespace

import numpy as np
import pytest

from emridispatch.config import INJECTION_KEYS
from emridispatch.parameters import (
    LOG_PARAMS, LOG_ROWS, NDIM, PARAM_NAMES, VECTOR_TO_PHYSICAL,
    physical_from_vector, truth_vector)

# Physical values for the keys with a domain worth respecting; anything else
# (including a parameter added later) gets a distinct negative filler, so no
# two keys can share a value and a permuted mapping cannot pass.
PHYSICAL = {"mass_1": 1.0e6, "mass_2": 10.0, "a": 0.3, "p": 11.0, "e": 0.1,
            "luminosity_distance": 1.5}
INJECTION = {name: PHYSICAL.get(name, -(0.35 + 0.13 * i))
             for i, name in enumerate(INJECTION_KEYS)}
VEC = 0.7 + 0.37 * np.arange(NDIM)

UNSAMPLED = sorted(set(INJECTION) - set(VECTOR_TO_PHYSICAL.values()))

needs_unsampled = pytest.mark.skipif(
    not UNSAMPLED, reason="every injection parameter is sampled")


def test_the_fiducial_covers_every_injection_key():
    assert set(INJECTION) == set(INJECTION_KEYS)
    assert len(set(INJECTION.values())) == len(INJECTION)


def test_every_sampled_row_has_an_injection_key():
    """A row added to PARAM_NAMES without a matching injection key would make
    truth_vector raise on every real config."""
    assert set(VECTOR_TO_PHYSICAL.values()) <= set(INJECTION_KEYS)


def test_every_log_parameter_is_sampled():
    """physical_from_vector exponentiates LOG_PARAMS unconditionally; one that
    is not a vector row would have its fiducial value exponentiated instead."""
    assert set(LOG_PARAMS) <= set(VECTOR_TO_PHYSICAL.values())


def test_vector_to_physical_covers_every_row():
    assert list(VECTOR_TO_PHYSICAL) == PARAM_NAMES
    assert len(set(VECTOR_TO_PHYSICAL.values())) == NDIM


def test_physical_from_vector_inverts_truth_vector():
    inj = physical_from_vector(VEC, INJECTION)
    assert np.allclose(truth_vector(inj), VEC)


def test_truth_vector_inverts_physical_from_vector():
    got = physical_from_vector(truth_vector(INJECTION), INJECTION)
    assert got == pytest.approx(INJECTION)


@needs_unsampled
def test_physical_from_vector_keeps_the_unsampled_parameters():
    inj = physical_from_vector(VEC, INJECTION)
    assert all(inj[name] == INJECTION[name] for name in UNSAMPLED)
    assert set(inj) == set(INJECTION)


def test_the_log_rows_lead_the_sampling_vector():
    """bounds.py's mass Jacobian (jac[0], jac[1]) and GridReparam.phi's
    b[:, 0] / b[:, 1] index them positionally, unlike the mapping helpers."""
    assert LOG_ROWS == (0, 1)


def test_the_masses_are_the_log_sampled_pair():
    """Anchors LOG_PARAMS itself. Both helpers derive their log handling from
    it, so the round-trip stays self-consistent even if it is wrong; only a
    named, physical assertion catches that."""
    vec = truth_vector(INJECTION)
    assert vec[PARAM_NAMES.index("ln_m1")] == pytest.approx(
        np.log(INJECTION["mass_1"]))
    assert vec[PARAM_NAMES.index("ln_m2")] == pytest.approx(
        np.log(INJECTION["mass_2"]))
    assert vec[PARAM_NAMES.index("p")] == pytest.approx(INJECTION["p"])


def test_physical_from_vector_exponentiates_only_the_log_rows():
    inj = physical_from_vector(VEC, INJECTION)
    for i, name in enumerate(PARAM_NAMES):
        want = np.exp(VEC[i]) if i in LOG_ROWS else VEC[i]
        assert inj[VECTOR_TO_PHYSICAL[name]] == pytest.approx(want), name


def test_physical_from_vector_does_not_mutate_the_fiducial():
    fiducial = dict(INJECTION)
    physical_from_vector(VEC, fiducial)
    assert fiducial == INJECTION


def test_physical_from_vector_returns_plain_floats():
    inj = physical_from_vector(np.asarray(VEC, dtype=np.float32), INJECTION)
    assert all(type(inj[VECTOR_TO_PHYSICAL[name]]) is float
               for name in PARAM_NAMES)


def test_workbench_to_physical_uses_the_shared_mapping():
    from emridispatch.workbench import to_physical

    model = SimpleNamespace(injection_parameters=INJECTION)
    assert to_physical(model, VEC) == physical_from_vector(
        VEC, {name: INJECTION[name] for name in UNSAMPLED})


def _stub_likelihood(injection, evaluate):
    """A real instance without __init__, which needs the full lisatools stack."""
    from emridispatch.response.lisatools import LisatoolsEMRILikelihood

    model = object.__new__(LisatoolsEMRILikelihood)
    model.injection_parameters = dict(injection)
    model.evaluate_likelihood = evaluate
    return model


def _capture_template(injection, vec):
    """The template dict __call__ hands to the waveform, without a waveform."""
    seen = {}
    _stub_likelihood(injection, lambda tp: seen.update(tp) or 0.0)(vec)
    return seen


def test_likelihood_template_uses_the_shared_mapping():
    """The three vector -> physical call sites must not drift apart."""
    assert _capture_template(INJECTION, VEC) == physical_from_vector(
        VEC, INJECTION)


@needs_unsampled
def test_likelihood_template_inherits_the_unsampled_parameters():
    """Inert for the current equatorial models, but a model that does use them
    must see the injection's values, not hard-coded ones."""
    injection = dict(INJECTION)
    for j, name in enumerate(UNSAMPLED):
        injection[name] = 100.0 + j
    template = _capture_template(injection, VEC)
    assert all(template[name] == injection[name] for name in UNSAMPLED)


def test_likelihood_rejects_a_short_vector():
    model = _stub_likelihood(INJECTION, lambda tp: 0.0)
    assert model(VEC[:-1]) == -np.inf


def test_pp_draw_truth_uses_the_shared_mapping():
    from emridispatch import pp
    from emridispatch.priors import joint_prior_from_box

    # Narrow box around the (valid) fiducial, so no draw is rejected whatever
    # the vector's length or row order.
    truth = truth_vector(INJECTION)
    prior = joint_prior_from_box(truth - 0.01, truth + 0.01, (),
                                 names=PARAM_NAMES)
    vec = prior.sample(np.random.default_rng(11))
    inj = pp.draw_truth(prior, np.random.default_rng(11), INJECTION)
    assert inj == physical_from_vector(vec, INJECTION)
