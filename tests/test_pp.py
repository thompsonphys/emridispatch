import json

import numpy as np
import pytest
from scipy.stats import kstest

from conftest import make_run_dir

from emridispatch import pp
from emridispatch.parameters import NDIM, PARAM_NAMES
from emridispatch.priors import joint_prior_from_box, joint_prior_from_config

MINS = np.array([13.0, 2.0, -0.5, 8.0, 0.05, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
MAXES = np.array([14.0, 2.6, 0.9, 12.0, 0.5, 2.0, np.pi, 2 * np.pi, np.pi,
                  2 * np.pi, 2 * np.pi, 2 * np.pi])
PERIODIC = (7, 9, 10, 11)
FIDUCIAL = {"x": 1.0, "phi_theta": 0.0, "mass_1": 1e6, "mass_2": 10.0}


def _prior(overrides=None):
    return joint_prior_from_box(MINS, MAXES, PERIODIC, names=PARAM_NAMES,
                                overrides=overrides)


def test_draw_truth_samples_the_configured_prior():
    """P-P is only valid when truths come from the prior the sampler uses."""
    prior = _prior({"q_s": {"type": "sine"}})
    rng = np.random.default_rng(0)
    q_s = np.array([pp.draw_truth(prior, rng, FIDUCIAL)["q_s"]
                    for _ in range(1500)])
    # Zero-information limit: the rank of a truth is the prior CDF at it, so
    # ranks are uniform iff the truths follow that prior.
    sine_cdf = (1.0 - np.cos(q_s)) / 2.0
    assert kstest(sine_cdf, "uniform").pvalue > 0.01


def test_draw_truth_is_uniform_without_overrides():
    prior = _prior()
    rng = np.random.default_rng(1)
    q_s = np.array([pp.draw_truth(prior, rng, FIDUCIAL)["q_s"]
                    for _ in range(1500)])
    assert kstest(q_s / np.pi, "uniform").pvalue > 0.01


def test_draw_truth_matches_the_prior_the_run_will_sample(toy_cfg):
    """pp and the pipeline must build the same JointPrior from the same box."""
    from emridispatch.pipeline import build_problem

    build_problem(toy_cfg, resume=False)
    with open(f"{toy_cfg.run.outdir}/prior_spec.json") as fh:
        run_spec = json.load(fh)
    box = np.load(f"{toy_cfg.run.outdir}/prior_bounds.npz")
    pp_spec = joint_prior_from_config(toy_cfg, box["mins"], box["maxes"]).spec()
    assert pp_spec == run_spec


def test_draw_truth_is_reproducible_under_a_seed():
    prior = _prior({"q_s": {"type": "sine"}})
    a = pp.draw_truth(prior, np.random.default_rng(5), FIDUCIAL)
    b = pp.draw_truth(prior, np.random.default_rng(5), FIDUCIAL)
    assert a == b


def test_draw_truth_keeps_the_fiducial_unsampled_parameters():
    inj = pp.draw_truth(_prior(), np.random.default_rng(3), FIDUCIAL)
    assert inj["x"] == FIDUCIAL["x"] and inj["phi_theta"] == FIDUCIAL["phi_theta"]


def test_draw_truth_rejects_waveform_invalid_draws():
    """A box straddling the e < 0.75 limit still yields only valid truths."""
    maxes = MAXES.copy()
    maxes[4] = 1.5
    prior = joint_prior_from_box(MINS, maxes, PERIODIC, names=PARAM_NAMES)
    rng = np.random.default_rng(2)
    for _ in range(200):
        assert pp.draw_truth(prior, rng, FIDUCIAL)["e"] <= 0.75


def test_draw_truth_raises_when_the_box_has_no_valid_point():
    mins, maxes = MINS.copy(), MAXES.copy()
    mins[4], maxes[4] = 0.9, 0.95          # e entirely above the 0.75 limit
    prior = joint_prior_from_box(mins, maxes, PERIODIC, names=PARAM_NAMES)
    with pytest.raises(RuntimeError, match="valid truth"):
        pp.draw_truth(prior, np.random.default_rng(4), FIDUCIAL, max_tries=50)


@pytest.mark.parametrize("idx,value,ok", [
    (2, 1.5, False), (2, 0.5, True),        # a
    (4, 0.9, False), (4, 0.5, True),        # e
    (5, 0.0, False), (5, 1.0, True),        # distance
])
def test_valid_truth_boundaries(idx, value, ok):
    vec = np.array([13.5, 2.3, 0.5, 10.0, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    vec[idx] = value
    assert pp._valid_truth(vec) is ok


def test_valid_truth_rejects_below_the_separatrix_floor():
    vec = np.array([13.5, 2.3, 0.5, 6.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert pp._valid_truth(vec) is False


@pytest.mark.parametrize("shift,expected", [(-50.0, 0.0), (50.0, 1.0)])
def test_compute_ranks_is_oriented_by_the_truth(tmp_path, shift, expected):
    """rank -> 1 when the truth sits above every sample, 0 when below.

    A flipped comparison stays uniform under the null, so KS cannot catch it.
    """
    d = make_run_dir(tmp_path / "run")
    truth = json.loads((d / "injection_truth.json").read_text())
    truth["sampling_vector"] = (np.full(NDIM, shift)).tolist()
    (d / "injection_truth.json").write_text(json.dumps(truth))
    ranks = pp.compute_ranks(str(d), burn_frac=0.25)
    assert all(r == expected for r in ranks.values())
