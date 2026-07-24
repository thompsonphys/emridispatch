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


def make_results(with_truth=True):
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((2, NSTEPS, NDIM))
    return Results(
        samples=samples,
        lnlike=rng.random((2, NSTEPS)),
        lnprob=rng.random((2, NSTEPS)),
        accepted=np.ones((2, NSTEPS)),
        temperatures=np.array([1.0, 2.5]),
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
    with pytest.raises(FileNotFoundError, match="emridispatch-postprocess"):
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
