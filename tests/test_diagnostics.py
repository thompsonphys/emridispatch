"""Backend-agnostic diagnostics: loading through the results layer, hybrid
R-hat (per run) / ESS (per walker series) views, burn in time steps."""

import numpy as np
import pytest

pytest.importorskip("h5py")

from conftest import (
    ERYN_IT, ERYN_NW, NSTEPS, make_eryn_run_dir, make_run_dir)
from emridispatch.diagnostics import load_cold_chain, stack_chains
from emridispatch.parameters import NDIM, PARAM_NAMES
from emridispatch.results import convert, is_complete

P0 = PARAM_NAMES[0]


def test_load_cold_chain_impulse_matches_raw(tmp_path):
    run = make_run_dir(tmp_path)
    s, ll = load_cold_chain(run)
    raw = np.loadtxt(run / "chain_0.txt")
    assert s.shape == (NSTEPS, NDIM)
    assert np.allclose(s, raw[:, :NDIM])
    assert np.allclose(ll, raw[:, NDIM])


def test_load_cold_chain_eryn_flattened(tmp_path):
    run = make_eryn_run_dir(tmp_path)
    s, ll = load_cold_chain(run)
    assert s.shape == (ERYN_IT * ERYN_NW, NDIM)
    assert ll.shape == (ERYN_IT * ERYN_NW,)
    # Step-major: index = step * nwalkers + walker; param 0 = step + 0.1*walker.
    assert np.isclose(s[0, 0], 0.0)
    assert np.isclose(s[1, 0], 0.1)
    assert np.isclose(s[ERYN_NW, 0], 1.0)


def test_stack_chains_hybrid_views(tmp_path):
    runs = [make_eryn_run_dir(tmp_path / "a"), make_eryn_run_dir(tmp_path / "b")]
    post, series = stack_chains(runs)
    # R-hat view: one chain per run dir, walkers never counted as chains.
    assert post["lnlike"].shape == (2, ERYN_IT * ERYN_NW)
    # Autocorrelation view: per-walker time series pooled over runs.
    assert series["lnlike"].shape == (2 * ERYN_NW, ERYN_IT)
    for w in range(ERYN_NW):
        assert np.allclose(series[P0][w], np.arange(ERYN_IT) + 0.1 * w)


def test_stack_chains_burn_is_time_steps(tmp_path):
    run = make_eryn_run_dir(tmp_path)
    post, series = stack_chains([run], burn=2)
    assert series["lnlike"].shape == (ERYN_NW, ERYN_IT - 2)
    assert post["lnlike"].shape == (1, (ERYN_IT - 2) * ERYN_NW)
    assert np.allclose(series[P0][0], np.arange(2, ERYN_IT))


def test_stack_chains_impulse_views_identical(tmp_path):
    run = make_run_dir(tmp_path)
    post, series = stack_chains([run], burn=10)
    assert post["lnlike"].shape == series["lnlike"].shape == (1, NSTEPS - 10)
    assert np.allclose(post[P0], series[P0])


def test_results_h5_preferred_over_raw(tmp_path):
    # The user contract: downstream reads the common results.h5, never raw
    # backend output. Delete the raw chains and diagnostics must still work.
    imp = make_run_dir(tmp_path / "imp")
    convert(imp).save(imp / "results.h5")
    (imp / "chain_0.txt").unlink()
    (imp / "chain_1.txt").unlink()
    s, _ = load_cold_chain(imp)
    assert s.shape == (NSTEPS, NDIM)

    er = make_eryn_run_dir(tmp_path / "er")
    convert(er).save(er / "results.h5")
    (er / "eryn_chain.h5").unlink()
    _, series = stack_chains([er])
    assert series["lnlike"].shape == (ERYN_NW, ERYN_IT)


def test_walker_series_does_not_depend_on_the_config(tmp_path):
    """The ensemble width comes from the results file, not from run metadata.

    A killed run has no run_summary.json, so config metadata falls back to the
    raw config.yaml where nwalkers is nested under sampler.eryn. The old lookup
    only probed a flat top-level key, silently collapsed to nwalkers=1, and
    handed the interleaved chain to the autocorrelation estimators as if it
    were a single time series.
    """
    run = make_eryn_run_dir(tmp_path)
    (run / "run_summary.json").unlink()
    res = convert(run)
    assert "nwalkers" not in res.config
    assert res.nwalkers == ERYN_NW

    res.save(run / "results.h5")
    _, series = stack_chains([run])
    assert series["lnlike"].shape == (ERYN_NW, ERYN_IT)
    for w in range(ERYN_NW):
        assert np.allclose(series[P0][w], np.arange(ERYN_IT) + 0.1 * w)


def test_stack_chains_mixed_backends_common_length(tmp_path):
    runs = [make_run_dir(tmp_path / "imp"), make_eryn_run_dir(tmp_path / "er")]
    post, series = stack_chains(runs)
    assert post["lnlike"].shape == (2, min(NSTEPS, ERYN_IT * ERYN_NW))
    assert series["lnlike"].shape == (1 + ERYN_NW, min(NSTEPS, ERYN_IT))


def test_is_complete_backend_agnostic(tmp_path):
    assert is_complete(make_run_dir(tmp_path / "imp"))
    # Regression: the old check required impulse's chain_0.txt.
    assert is_complete(make_eryn_run_dir(tmp_path / "er"))
    assert not is_complete(make_run_dir(tmp_path / "inc", with_summary=False))


def test_report_smoke(tmp_path, capsys):
    pytest.importorskip("arviz")
    pytest.importorskip("emcee")
    from emridispatch.diagnostics import report

    runs = [make_run_dir(tmp_path / "a"), make_run_dir(tmp_path / "b")]
    summ = report(runs, burn=10)
    out = capsys.readouterr().out
    assert "cold runs: 2" in out
    assert "r_hat" in summ.columns


def test_report_eryn_single_run_no_walker_rhat(tmp_path, capsys):
    pytest.importorskip("arviz")
    pytest.importorskip("emcee")
    from emridispatch.diagnostics import report

    report([make_eryn_run_dir(tmp_path)])
    out = capsys.readouterr().out
    # Coupled walkers must not count as independent chains for R-hat.
    assert "R-hat needs >= 2 independent cold runs" in out
    assert "walker series" in out
