from types import SimpleNamespace

import numpy as np
import pytest

from emridispatch.parameters import (
    NDIM, PARAM_NAMES, VECTOR_TO_PHYSICAL, physical_from_vector, truth_vector)

INJECTION = {
    "mass_1": 1.0e6, "mass_2": 10.0, "a": 0.3, "p": 10.0, "e": 0.1, "x": 1.0,
    "luminosity_distance": 1.5, "q_s": 0.8, "phi_s": 1.2, "q_k": 1.9,
    "phi_k": 2.4, "phi_phi": 3.1, "phi_theta": 0.7, "phi_r": 5.5,
}
VEC = np.array([13.5, 2.4, -0.2, 11.0, 0.25, 2.5,
                0.9, 1.1, 2.0, 3.0, 4.0, 5.0])


def test_vector_to_physical_covers_every_row():
    assert list(VECTOR_TO_PHYSICAL) == PARAM_NAMES
    assert len(set(VECTOR_TO_PHYSICAL.values())) == NDIM


def test_physical_from_vector_inverts_truth_vector():
    inj = physical_from_vector(VEC, INJECTION)
    assert np.allclose(truth_vector(inj), VEC)


def test_truth_vector_inverts_physical_from_vector():
    got = physical_from_vector(truth_vector(INJECTION), INJECTION)
    assert got == pytest.approx(INJECTION)


def test_physical_from_vector_keeps_the_unsampled_parameters():
    inj = physical_from_vector(VEC, INJECTION)
    assert inj["x"] == INJECTION["x"]
    assert inj["phi_theta"] == INJECTION["phi_theta"]
    assert set(inj) == set(INJECTION)


def test_physical_from_vector_exponentiates_only_the_mass_rows():
    inj = physical_from_vector(VEC, INJECTION)
    assert inj["mass_1"] == pytest.approx(np.exp(VEC[0]))
    assert inj["mass_2"] == pytest.approx(np.exp(VEC[1]))
    assert inj["a"] == pytest.approx(VEC[2])
    assert inj["p"] == pytest.approx(VEC[3])


def test_physical_from_vector_does_not_mutate_the_fiducial():
    fiducial = dict(INJECTION)
    physical_from_vector(VEC, fiducial)
    assert fiducial == INJECTION


def test_physical_from_vector_returns_plain_floats():
    inj = physical_from_vector(VEC, INJECTION)
    assert all(type(v) is float for v in inj.values())


def test_workbench_to_physical_uses_the_shared_mapping():
    from emridispatch.workbench import to_physical

    model = SimpleNamespace(injection_parameters=INJECTION)
    assert to_physical(model, VEC) == physical_from_vector(
        VEC, {"x": INJECTION["x"], "phi_theta": INJECTION["phi_theta"]})


def test_likelihood_template_uses_the_shared_mapping():
    """The three vector -> physical call sites must not drift apart."""
    from emridispatch.response.lisatools import LisatoolsEMRILikelihood

    seen = {}
    stub = SimpleNamespace(
        default_parameters=dict(INJECTION),
        evaluate_likelihood=lambda tp: seen.update(tp) or 0.0)
    LisatoolsEMRILikelihood.__call__(stub, VEC)
    assert seen == physical_from_vector(VEC, INJECTION)


def test_likelihood_rejects_a_short_vector():
    from emridispatch.response.lisatools import LisatoolsEMRILikelihood

    stub = SimpleNamespace(default_parameters=dict(INJECTION),
                           evaluate_likelihood=lambda tp: 0.0)
    assert LisatoolsEMRILikelihood.__call__(stub, VEC[:5]) == -np.inf


def test_pp_draw_truth_uses_the_shared_mapping():
    from emridispatch import pp
    from emridispatch.priors import joint_prior_from_box

    mins = np.array([13.0, 2.0, -0.5, 9.0, 0.05, 0.5, 0, 0, 0, 0, 0, 0])
    maxes = np.array([14.0, 2.6, 0.9, 12.0, 0.5, 2.0, np.pi, 2 * np.pi,
                      np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi])
    prior = joint_prior_from_box(mins, maxes, (7, 9, 10, 11), names=PARAM_NAMES)
    vec = prior.sample(np.random.default_rng(11))
    inj = pp.draw_truth(prior, np.random.default_rng(11), INJECTION)
    assert inj == physical_from_vector(vec, INJECTION)
