"""Plotting tests: results.h5 -> corner.png + marginals.png."""

import sys

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("matplotlib")
pytest.importorskip("corner")

from emridispatch.parameters import NDIM, PARAM_NAMES
from emridispatch.results import Results
from emridispatch.viz import make_plots, resolve_results_path
from emridispatch.viz import main as plot_main

NSTEPS = 150


TEMPS = np.array([1.0, 2.5])


def make_results(with_truth=True, betas=None):
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((2, NSTEPS, NDIM))
    lnlike = rng.random((2, NSTEPS))
    lnprior = -np.ones((2, NSTEPS))
    if betas is None:
        betas = np.tile(1.0 / TEMPS, (NSTEPS, 1))
    return Results(
        samples=samples,
        lnlike=lnlike,
        lnprior=lnprior,
        lnprob=lnprior + betas.T * lnlike,
        accepted=np.ones((2, NSTEPS)),
        betas=betas,
        temperatures=1.0 / betas[-1],
        param_names=list(PARAM_NAMES),
        physical=samples + 1.0,
        truth_physical=np.ones(NDIM) if with_truth else None,
        truth_sampling=np.zeros(NDIM) if with_truth else None,
        backend="impulse",
    )


def test_make_plots_default(tmp_path):
    written = make_plots(make_results(), outdir=str(tmp_path))
    names = sorted(p.split("/")[-1] for p in written)
    assert names == ["corner.png", "marginals.png"]
    for p in written:
        assert (tmp_path / p.split("/")[-1]).stat().st_size > 0


def test_rung_legend_reports_a_range_when_the_ladder_moved():
    from emridispatch.viz import rung_legend

    fixed = make_results()
    assert rung_legend(fixed, 0) == "$T$ = 1"
    assert rung_legend(fixed, 1) == "$T$ = 2.5"

    drifting = np.tile(1.0 / TEMPS, (NSTEPS, 1))
    drifting[:, 1] = np.linspace(1 / 2.0, 1 / 4.0, NSTEPS)
    adapted = make_results(betas=drifting)
    assert adapted.ladder_adapted()
    # The cold rung is pinned at beta=1, so it keeps a single label.
    assert rung_legend(adapted, 0) == "$T$ = 1"
    assert rung_legend(adapted, 1) == "$T$ = 2\N{EN DASH}4"


def test_make_plots_all_temps(tmp_path):
    written = make_plots(make_results(), all_temps=True, outdir=str(tmp_path))
    assert len(written) == 2


def test_make_plots_no_truth(tmp_path):
    written = make_plots(make_results(with_truth=False), outdir=str(tmp_path))
    assert len(written) == 2


def test_burn_guard(tmp_path):
    with pytest.raises(ValueError, match="reduce --burn"):
        make_plots(make_results(), burn=NSTEPS, outdir=str(tmp_path))


def test_rung_out_of_range(tmp_path):
    with pytest.raises(ValueError, match="out of range"):
        make_plots(make_results(), temps=[5], outdir=str(tmp_path))


def test_resolve_results_path(tmp_path):
    res = make_results()
    h5 = tmp_path / "results.h5"
    res.save(h5)
    assert resolve_results_path(str(tmp_path)) == str(h5)
    assert resolve_results_path(str(h5)) == str(h5)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="emridisp-postprocess"):
        resolve_results_path(str(empty))


def test_cli_smoke(tmp_path, monkeypatch, capsys):
    make_results().save(tmp_path / "results.h5")
    out = tmp_path / "plots"
    monkeypatch.setattr(sys, "argv", [
        "emridispatch-plot", str(tmp_path), "--outdir", str(out),
        "--burn", "10", "--all-temps"])
    plot_main()
    assert (out / "corner.png").exists()
    assert (out / "marginals.png").exists()
    assert "wrote" in capsys.readouterr().out
