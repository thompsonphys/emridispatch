"""Eryn backend: eryn-free adapter unit tests + toy end-to-end run.

The adapter classes are pure numpy, so most of this file runs without eryn;
the end-to-end/resume tests are skipped when eryn is not installed.
"""

import os

import numpy as np
import pytest

from emridispatch.backends.eryn import CHAIN_NAME, _initial_walkers, _WhitenedPrior
from emridispatch.parameters import NDIM
from emridispatch.priors import JointPrior, Uniform


def make_prior(ndim=4):
    return JointPrior([Uniform(-1.0, 1.0) for _ in range(ndim)],
                      names=[f"p{i}" for i in range(ndim)])


def test_whitened_prior_logpdf_batched():
    joint = make_prior()
    adapter = _WhitenedPrior(joint, 4)
    x = np.array([[0.0, 0.0, 0.0, 0.0],
                  [0.5, -0.5, 0.2, 0.9],
                  [2.0, 0.0, 0.0, 0.0]])   # last row out of bounds
    out = adapter.logpdf(x)
    assert out.shape == (3,)
    assert np.isclose(out[0], joint(x[0]))
    assert np.isclose(out[1], joint(x[1]))
    assert out[2] == -np.inf
    # 1-D input promotes to a single-row batch.
    assert np.isclose(adapter.logpdf(x[0])[0], joint(x[0]))


def test_whitened_prior_key_order_scalar():
    # Must be a scalar: eryn's resume path compares key_order dicts after an
    # h5-attr round trip, and array-valued entries make that comparison raise.
    adapter = _WhitenedPrior(make_prior(), 4)
    assert isinstance(adapter.key_order, int)


def test_initial_walkers_shape_and_validity():
    adapter = _WhitenedPrior(make_prior(), 4)
    coords = _initial_walkers(np.zeros(4), np.eye(4) * 0.01, adapter,
                              3, 10, np.random.default_rng(0))
    assert coords.shape == (3, 10, 4)
    assert np.all(np.isfinite(adapter.logpdf(coords.reshape(-1, 4))))


def test_initial_walkers_pins_stuck_rows_to_start():
    adapter = _WhitenedPrior(make_prior(), 4)
    # Huge covariance: essentially every draw lands outside the prior box, so
    # rows fall back to the (finite-prior) start point.
    coords = _initial_walkers(np.zeros(4), np.eye(4) * 1e6, adapter,
                              1, 4, np.random.default_rng(0), max_tries=2)
    assert np.all(np.isfinite(adapter.logpdf(coords.reshape(-1, 4))))


def test_eryn_backend_end_to_end_and_resume(toy_cfg):
    pytest.importorskip("eryn")
    import h5py

    from emridispatch.backends import get_backend
    from emridispatch.pipeline import build_problem

    toy_cfg.sampler.backend = "eryn"
    toy_cfg.sampler.nsamples = 20
    toy_cfg.sampler.eryn.nwalkers = 30
    toy_cfg.sampler.eryn.ntemps = 2
    problem = build_problem(toy_cfg)
    summary = get_backend("eryn").run(problem, toy_cfg, resume=False)
    assert summary is not None
    assert summary["config"]["backend"] == "eryn"
    assert summary["config"]["nwalkers"] == 30

    chain_path = os.path.join(problem.outdir, CHAIN_NAME)
    assert os.path.exists(chain_path)
    assert os.path.exists(os.path.join(problem.outdir, "run_summary.json"))
    with h5py.File(chain_path, "r") as f:
        assert int(f["mcmc"].attrs["iteration"]) == 20
        assert f["mcmc/chain/model_0"].shape[1:] == (2, 30, 1, NDIM)

    # Resume appends to the same file (exercises the scalar key_order
    # round trip through the h5 attrs).
    summary2 = get_backend("eryn").run(problem, toy_cfg, resume=True)
    assert summary2 is not None
    with h5py.File(chain_path, "r") as f:
        assert int(f["mcmc"].attrs["iteration"]) == 40


def test_eryn_backend_rejects_unknown_move(toy_cfg):
    pytest.importorskip("eryn")
    from emridispatch.backends import get_backend
    from emridispatch.pipeline import build_problem

    toy_cfg.sampler.backend = "eryn"
    toy_cfg.sampler.eryn.move = "gaussian"
    problem = build_problem(toy_cfg)
    with pytest.raises(ValueError, match="sampler.eryn.move"):
        get_backend("eryn").run(problem, toy_cfg, resume=False)
