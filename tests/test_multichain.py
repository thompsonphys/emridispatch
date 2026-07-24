"""Multichain driver: backend-agnostic completion + per-seed results.h5."""

import sys
import textwrap

import pytest

pytest.importorskip("h5py")

import yaml

from conftest import MINIMAL_TOY_CONFIG, make_eryn_run_dir, make_run_dir
from emridispatch import multichain
from emridispatch.results import Results, is_complete


def test_ensure_results_writes_and_is_idempotent(tmp_path):
    for name, make in (("imp", make_run_dir), ("er", make_eryn_run_dir)):
        run = make(tmp_path / name)
        multichain._ensure_results(str(run))
        path = run / "results.h5"
        assert path.exists()
        assert Results.load(path).samples.ndim == 3
        mtime = path.stat().st_mtime_ns
        multichain._ensure_results(str(run))
        assert path.stat().st_mtime_ns == mtime  # existing file untouched


def _write_toy_yaml(tmp_path, backend="impulse"):
    raw = yaml.safe_load(textwrap.dedent(
        MINIMAL_TOY_CONFIG.format(outdir=str(tmp_path / "chains"))))
    raw["sampler"]["backend"] = backend
    if backend == "eryn":
        # Stretch move needs >= 2*ndim walkers.
        raw["sampler"]["eryn"] = {"nwalkers": 30, "ntemps": 1}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def _run_multichain(monkeypatch, cfg_path, mc_root):
    monkeypatch.setattr(sys, "argv", [
        "emridispatch-multichain", str(cfg_path), "--outdir", str(mc_root),
        "--nchains", "2", "--nsamples", "60", "--start-mode", "truth",
        "--no-report"])
    multichain.main()


def _assert_seeds_done(mc_root, backend):
    for seed in (42, 43):
        d = mc_root / f"seed_{seed}"
        assert is_complete(d)
        assert (d / "results.h5").exists()
        res = Results.load(d / "results.h5")
        assert res.backend == backend
        assert res.nsteps > 1


def test_multichain_toy_impulse(tmp_path, monkeypatch):
    pytest.importorskip("impulse")
    cfg_path = _write_toy_yaml(tmp_path, backend="impulse")
    mc_root = tmp_path / "mc"
    _run_multichain(monkeypatch, cfg_path, mc_root)
    _assert_seeds_done(mc_root, "impulse")


def test_multichain_toy_eryn(tmp_path, monkeypatch):
    # Regression: the old is_complete required chain_0.txt, so every eryn
    # seed was marked FAILED even after a successful run.
    pytest.importorskip("eryn")
    cfg_path = _write_toy_yaml(tmp_path, backend="eryn")
    mc_root = tmp_path / "mc"
    _run_multichain(monkeypatch, cfg_path, mc_root)
    _assert_seeds_done(mc_root, "eryn")
