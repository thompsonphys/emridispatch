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


def test_eryn_backend_acceptance_rates_ignore_thin_by(toy_cfg):
    """Rates must divide by stored steps, not proposals.

    eryn rebuilds its per-step `accepted` array inside the thin_by inner loop
    and hands save_step only the final one, so both the proposal and swap
    counters advance once per stored step. Dividing by iteration*thin_by
    understated every rate by exactly thin_by, and disagreed with the
    independent normalization in results._convert_eryn.
    """
    pytest.importorskip("eryn")
    h5py = pytest.importorskip("h5py")

    from emridispatch.backends import get_backend
    from emridispatch.pipeline import build_problem
    from emridispatch.results import convert

    thin_by, nwalkers, nsamples = 3, 30, 8
    toy_cfg.sampler.backend = "eryn"
    toy_cfg.sampler.nsamples = nsamples
    toy_cfg.sampler.eryn.nwalkers = nwalkers
    toy_cfg.sampler.eryn.ntemps = 2
    toy_cfg.sampler.eryn.thin_by = thin_by
    problem = build_problem(toy_cfg)
    summary = get_backend("eryn").run(problem, toy_cfg, resume=False)

    with h5py.File(os.path.join(problem.outdir, CHAIN_NAME), "r") as f:
        it = int(f["mcmc"].attrs["iteration"])
        raw_accepted = f["mcmc/accepted"][()]
        raw_swaps = f["mcmc/swaps_accepted"][()]
    assert it == nsamples
    assert raw_accepted.max() <= it

    rates = np.asarray(summary["proposal_acceptance"])
    assert rates.shape == (2,)
    assert np.all((rates > 0.0) & (rates <= 1.0))
    assert np.allclose(rates, (raw_accepted / it).mean(axis=1))

    swaps = np.asarray(summary["swap_acceptance"])
    assert swaps.shape == (1,)
    assert np.all((swaps >= 0.0) & (swaps <= 1.0))
    assert np.allclose(swaps, raw_swaps / (it * nwalkers))

    # The converter normalizes the same counters independently; the two
    # must not drift apart again.
    res = convert(problem.outdir)
    assert np.allclose(rates, res.accepted.mean(axis=1))


def test_eryn_backend_rejects_too_few_walkers(toy_cfg):
    pytest.importorskip("eryn")
    from emridispatch.backends import get_backend
    from emridispatch.pipeline import build_problem

    toy_cfg.sampler.backend = "eryn"
    toy_cfg.sampler.eryn.nwalkers = 2 * NDIM - 2
    problem = build_problem(toy_cfg)
    with pytest.raises(ValueError, match=f"nwalkers.*{2 * NDIM}"):
        get_backend("eryn").run(problem, toy_cfg, resume=False)


def test_eryn_backend_accepts_exactly_twice_ndim_walkers(toy_cfg):
    pytest.importorskip("eryn")
    from emridispatch.backends import get_backend
    from emridispatch.pipeline import build_problem

    toy_cfg.sampler.backend = "eryn"
    toy_cfg.sampler.nsamples = 2
    toy_cfg.sampler.eryn.nwalkers = 2 * NDIM
    problem = build_problem(toy_cfg)
    summary = get_backend("eryn").run(problem, toy_cfg, resume=False)
    assert summary is not None


def test_eryn_backend_rejects_unknown_move(toy_cfg):
    pytest.importorskip("eryn")
    from emridispatch.backends import get_backend
    from emridispatch.pipeline import build_problem

    toy_cfg.sampler.backend = "eryn"
    toy_cfg.sampler.eryn.move = "gaussian"
    problem = build_problem(toy_cfg)
    with pytest.raises(ValueError, match="sampler.eryn.move"):
        get_backend("eryn").run(problem, toy_cfg, resume=False)
