"""End-to-end structure tests with the toy Gaussian model + heuristic Fisher.

build_problem needs only numpy/scipy/pyyaml; the impulse-backend run test is
skipped when impulse is not installed.
"""

import os

import numpy as np
import pytest

from emridispatch.parameters import NDIM, PARAM_NAMES
from emridispatch.pipeline import build_problem
from emridispatch.priors import JointPrior


def test_build_problem_toy(toy_cfg):
    problem = build_problem(toy_cfg)

    assert problem.ndim == NDIM
    assert problem.param_names == PARAM_NAMES
    assert isinstance(problem.prior, JointPrior)
    assert problem.periodic == {i: 2 * np.pi for i in (7, 9, 10, 11)}
    assert problem.whitened  # reparam mode auto

    # Physical-space pieces are consistent.
    assert problem.x0.shape == (NDIM,)
    assert np.all(problem.x0 >= problem.prior.mins)
    assert np.all(problem.x0 <= problem.prior.maxes)
    assert np.isfinite(problem.prior(problem.x0))
    assert np.isfinite(problem.lnlike(problem.x0))
    # Truth start mode: x0 == truth, and toy lnL peaks there.
    assert np.allclose(problem.x0, problem.truth)
    assert problem.lnlike(problem.truth) == 0.0

    # Outdir artifacts written.
    assert os.path.exists(os.path.join(problem.outdir, "injection_truth.json"))
    assert os.path.exists(os.path.join(problem.outdir, "prior_bounds.npz"))
    assert os.path.exists(os.path.join(problem.outdir, "reparam_transform.npz"))


def test_wrapped_view_consistency(toy_cfg):
    problem = build_problem(toy_cfg)
    w = problem.wrapped()

    # Wrapped callables at the transformed start match the physical evaluation.
    assert np.isclose(w.lnlike(w.x0), problem.lnlike(problem.x0))
    assert np.isclose(w.lnprior(w.x0), problem.prior(problem.x0))
    # Round trip through the reparam.
    assert np.allclose(problem.reparam.to_x(w.x0), problem.x0)
    assert w.proposal_cov.shape == (NDIM, NDIM)
    assert w.periodic == problem.periodic


def test_build_problem_resume_uses_cache(toy_cfg):
    p1 = build_problem(toy_cfg)
    # Second build resumes off the cached bounds -- identical box.
    p2 = build_problem(toy_cfg, resume=True)
    assert np.allclose(p1.prior.mins, p2.prior.mins)
    assert np.allclose(p1.prior.maxes, p2.prior.maxes)
    assert np.allclose(p1.proposal_cov, p2.proposal_cov)


def test_prior_overrides_reach_problem(toy_cfg):
    toy_cfg.priors = {"q_s": {"type": "sine"}, "q_k": {"type": "sine"}}
    problem = build_problem(toy_cfg)
    from emridispatch.priors import Sine

    assert isinstance(problem.prior["q_s"], Sine)
    assert isinstance(problem.prior["q_k"], Sine)
    # Non-uniform prior evaluates finitely at the start.
    assert np.isfinite(problem.prior(problem.x0))


def test_manual_sigmas_box_reaches_problem(toy_cfg):
    sigmas = {"mass_1": 1.0, "mass_2": 1e-4, "a": 1e-5,
              "p": 1e-6, "e": 1e-6, "luminosity_distance": 0.05}
    toy_cfg.prior.fisher = "manual"
    toy_cfg.prior.box_scale = 25.0
    toy_cfg.prior.sigmas = sigmas
    toy_cfg.priors = {"q_s": {"type": "sine"}}
    problem = build_problem(toy_cfg)
    # Rectangular box: truth +/- 25*sigma (masses in log; distance is SNR-
    # rescaled so only shape-invariant rows are asserted).
    assert np.isclose(np.exp(problem.prior.mins[0]), 1e6 - 25.0 * 1.0)
    assert np.isclose(np.exp(problem.prior.maxes[0]), 1e6 + 25.0 * 1.0)
    assert np.isclose(problem.prior.maxes[4], 0.1 + 25.0 * 1e-6)
    assert np.isclose(problem.prior.maxes[6], np.pi)
    # diag(sigma^2) proposal for the non-mass intrinsics.
    assert np.isclose(problem.proposal_cov[2, 2], 1e-5 ** 2)
    # Per-parameter overrides still compose on top of the box.
    from emridispatch.priors import Sine

    assert isinstance(problem.prior["q_s"], Sine)


def test_log_versions(toy_cfg, caplog):
    from emridispatch.pipeline import log_versions

    with caplog.at_level("INFO", logger="emridispatch.pipeline"):
        log_versions(toy_cfg)
    text = caplog.text
    assert "emridispatch=" in text
    assert "sampler backend" in text
    assert "response 'toy'" in text
    assert "fisher 'none'" in text  # conftest toy config: prior.fisher = none


def test_unknown_backend_errors(toy_cfg):
    from emridispatch.backends import get_backend

    with pytest.raises(ValueError, match="unknown sampler backend"):
        get_backend("nope")


def test_impulse_backend_end_to_end(toy_cfg):
    pytest.importorskip("impulse")
    from emridispatch.backends import get_backend

    problem = build_problem(toy_cfg)
    summary = get_backend("impulse").run(problem, toy_cfg, resume=False)
    assert summary is not None
    assert os.path.exists(os.path.join(problem.outdir, "run_summary.json"))
    assert os.path.exists(os.path.join(problem.outdir, "chain_0.txt"))


def test_a_fresh_start_removes_rungs_the_new_ladder_does_not_write(toy_cfg):
    """impulse truncates only the rungs it is handed, so a shorter ladder run
    into the same outdir would leave the old top rungs for the converter to
    read back as genuine ones."""
    pytest.importorskip("impulse")
    from emridispatch.backends import get_backend

    problem = build_problem(toy_cfg)
    orphan = os.path.join(problem.outdir, "chain_99.txt")
    os.makedirs(problem.outdir, exist_ok=True)
    with open(orphan, "w") as fh:
        fh.write("0.0\n")
    get_backend("impulse").run(problem, toy_cfg, resume=False)
    assert not os.path.exists(orphan)


def test_a_fresh_start_removes_the_stale_checkpoint(toy_cfg):
    """impulse reloads sampler_checkpoint.pkl wholesale (`__dict__.update`), so
    one left behind resurrects the abandoned run's ladder and iteration count
    on the next default-resume run, which then samples nothing. save_freq is
    raised to nsamples so this run writes no checkpoint of its own."""
    pytest.importorskip("impulse")
    from emridispatch.backends import get_backend

    toy_cfg.sampler.impulse.save_freq = toy_cfg.sampler.nsamples
    problem = build_problem(toy_cfg)
    ckpt = os.path.join(problem.outdir, "sampler_checkpoint.pkl")
    os.makedirs(problem.outdir, exist_ok=True)
    with open(ckpt, "wb") as fh:
        fh.write(b"stale")
    get_backend("impulse").run(problem, toy_cfg, resume=False)
    assert not os.path.exists(ckpt)
