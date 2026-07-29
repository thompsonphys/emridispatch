import json

import numpy as np
import pytest
from scipy.stats import kstest

from conftest import make_run_dir

from emridispatch import pp
from emridispatch.parameters import NDIM, PARAM_NAMES
from emridispatch.priors import joint_prior_from_box, joint_prior_from_config

# Built by name, so a row added to the sampling vector gets a usable default
# rather than an IndexError.
_BOX = {"ln_m1": (13.0, 14.0), "ln_m2": (2.0, 2.6), "a": (-0.5, 0.9),
        "p": (8.0, 12.0), "e": (0.05, 0.5), "dist": (0.5, 2.0),
        "q_s": (0.0, np.pi), "q_k": (0.0, np.pi)}


def _row(name):
    return PARAM_NAMES.index(name)


def _box():
    mins, maxes = np.zeros(NDIM), np.full(NDIM, 2 * np.pi)
    for name, (lo, hi) in _BOX.items():
        mins[_row(name)], maxes[_row(name)] = lo, hi
    return mins, maxes


MINS, MAXES = _box()
PERIODIC = tuple(_row(n) for n in ("phi_s", "phi_k", "phi_phi", "phi_r"))
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
    maxes[_row("e")] = 1.5
    prior = joint_prior_from_box(MINS, maxes, PERIODIC, names=PARAM_NAMES)
    rng = np.random.default_rng(2)
    for _ in range(200):
        assert pp.draw_truth(prior, rng, FIDUCIAL)["e"] <= 0.75


def test_draw_truth_raises_when_the_box_has_no_valid_point():
    mins, maxes = MINS.copy(), MAXES.copy()
    mins[_row("e")], maxes[_row("e")] = 0.9, 0.95   # e above the 0.75 limit
    prior = joint_prior_from_box(mins, maxes, PERIODIC, names=PARAM_NAMES)
    with pytest.raises(RuntimeError, match="valid truth"):
        pp.draw_truth(prior, np.random.default_rng(4), FIDUCIAL, max_tries=50)


def _valid_vec(**overrides):
    vec = np.ones(NDIM)
    vec[PARAM_NAMES.index("ln_m1")] = 13.5
    vec[PARAM_NAMES.index("ln_m2")] = 2.3
    vec[PARAM_NAMES.index("a")] = 0.5
    vec[PARAM_NAMES.index("p")] = 10.0
    vec[PARAM_NAMES.index("e")] = 0.1
    for name, value in overrides.items():
        vec[PARAM_NAMES.index(name)] = value
    return vec


@pytest.mark.parametrize("name,value,ok", [
    ("a", 1.5, False), ("a", 0.5, True),
    ("e", 0.9, False), ("e", 0.5, True),
    ("dist", 0.0, False), ("dist", 1.0, True),
])
def test_valid_truth_boundaries(name, value, ok):
    assert pp._valid_truth(_valid_vec(**{name: value})) is ok


def test_valid_truth_rejects_below_the_separatrix_floor():
    assert pp._valid_truth(_valid_vec(p=6.1)) is False


@pytest.mark.parametrize("name", ["a", "p", "e", "dist"])
def test_valid_truth_rejects_nan(name):
    """The separatrix floor must be written as `p > floor`, not `if p <= floor:
    reject` -- the latter passes NaN straight through into an injection."""
    assert pp._valid_truth(_valid_vec(**{name: np.nan})) is False


def test_validity_limits_name_real_parameters():
    """Rows are resolved from PARAM_NAMES, so a renamed or reordered vector
    must fail here rather than silently checking the wrong rows."""
    assert set(pp.VALIDITY) <= set(PARAM_NAMES)
    assert set(pp._VALIDITY_ROWS) == {PARAM_NAMES.index(n) for n in pp.VALIDITY}


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
