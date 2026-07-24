"""Converter + HDF5 round-trip tests for the common results format."""

import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("h5py")

from conftest import (
    ERYN_IT, ERYN_NW, ERYN_NT, NSTEPS, TEMPS, TRUTH,
    make_eryn_run_dir, make_run_dir)
from emridispatch.parameters import NDIM, PARAM_NAMES
from emridispatch.priors import (
    CallablePrior, Gaussian, JointPrior, Sine, Uniform, joint_prior_from_specs)
from emridispatch.results import Results, convert, detect_backend
from emridispatch.results import main as postprocess_main


def test_convert_shapes_and_truth(tmp_path):
    run = make_run_dir(tmp_path)
    res = convert(run)
    assert res.samples.shape == (2, NSTEPS, NDIM)
    assert res.lnlike.shape == (2, NSTEPS)
    assert np.allclose(res.temperatures, TEMPS)
    # No reparam file -> physical frame equals sampling frame, truth matches.
    assert np.allclose(res.physical, res.samples)
    assert np.allclose(res.truth_physical, TRUTH)
    assert np.allclose(res.truth_sampling, TRUTH)
    assert res.backend == "impulse"
    assert res.config["ntemps"] == 2


def test_convert_eryn_shapes_and_flattening(tmp_path):
    run = make_eryn_run_dir(tmp_path)
    assert detect_backend(run) == "eryn"
    res = convert(run)
    assert res.backend == "eryn"
    # Walkers flattened into the common schema; stale tail beyond
    # `iteration` sliced away.
    assert res.samples.shape == (ERYN_NT, ERYN_IT * ERYN_NW, NDIM)
    assert res.lnlike.shape == (ERYN_NT, ERYN_IT * ERYN_NW)
    # Step-major ordering: index = step * nwalkers + walker.
    for step in range(ERYN_IT):
        for wkr in range(ERYN_NW):
            assert np.isclose(res.samples[0, step * ERYN_NW + wkr, 0],
                              step + 0.1 * wkr)
    # lnprob = log_like + log_prior; log_prior was -1 everywhere.
    assert np.allclose(res.lnprob, res.lnlike - 1.0)
    # Temperatures from the last betas row; beta=0 -> inf.
    assert np.isclose(res.temperatures[0], 1.0)
    assert np.isinf(res.temperatures[1])
    # Per-walker acceptance fractions broadcast across steps.
    assert np.isclose(res.accepted[0].mean(), np.mean([1, 2, 3]) / ERYN_IT)
    assert np.isclose(res.accepted[1].mean(), np.mean([4, 5, 0]) / ERYN_IT)
    # Shared sidecars: truth + config metadata work as for impulse.
    assert np.allclose(res.truth_physical, TRUTH)
    assert res.config["backend"] == "eryn"


def test_eryn_save_load_roundtrip(tmp_path):
    run = make_eryn_run_dir(tmp_path)
    res = convert(run)
    path = tmp_path / "results.h5"
    res.save(path)
    back = Results.load(path)
    assert back.backend == "eryn"
    assert np.allclose(back.samples, res.samples)
    assert np.allclose(back.temperatures[0], 1.0)
    assert np.isinf(back.temperatures[1])
    assert json.loads(back.run_summary)["config"]["backend"] == "eryn"


def test_save_load_roundtrip(tmp_path):
    run = make_run_dir(tmp_path, with_spec=True)
    res = convert(run)
    path = tmp_path / "results.h5"
    res.save(path)
    back = Results.load(path)
    assert np.allclose(back.samples, res.samples)
    assert np.allclose(back.lnlike, res.lnlike)
    assert np.allclose(back.temperatures, res.temperatures)
    assert back.param_names == list(PARAM_NAMES)
    assert back.injection == res.injection
    assert back.prior_spec == res.prior_spec
    # Verbatim run files embedded for reproducibility.
    assert "backend: impulse" in back.config_yaml
    assert "versions:" in back.run_log
    assert json.loads(back.run_summary)["config"]["backend"] == "impulse"


def test_load_rejects_wrong_version(tmp_path):
    import h5py

    path = tmp_path / "bogus.h5"
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = 999
    with pytest.raises(ValueError, match="format_version"):
        Results.load(path)


def test_backend_detection_and_unknown(tmp_path):
    run = make_run_dir(tmp_path)
    assert detect_backend(run) == "impulse"
    with pytest.raises(ValueError, match="no converter"):
        convert(run, backend="nessai")


def test_missing_summary_needs_explicit_backend(tmp_path):
    run = make_run_dir(tmp_path, with_summary=False)
    (run / "config.yaml").unlink()  # neither summary nor config -> no detect
    with pytest.raises(ValueError, match="--backend"):
        convert(run)
    res = convert(run, backend="impulse")
    assert res.samples.shape == (2, NSTEPS, NDIM)
    assert res.config == {}


def test_missing_summary_falls_back_to_config_yaml(tmp_path):
    run = make_run_dir(tmp_path, with_summary=False)
    (run / "config.yaml").write_text("run:\n  seed: 7\nsampler:\n  backend: impulse\n")
    assert detect_backend(run) == "impulse"
    res = convert(run)  # auto-detect via config.yaml, no --backend needed
    assert res.config["run"]["seed"] == 7
    # config.yaml without a sampler section -> pipeline's impulse default.
    (run / "config.yaml").write_text("run:\n  seed: 7\n")
    assert detect_backend(run) == "impulse"


def test_prior_spec_roundtrip(tmp_path):
    priors = [Uniform(0.0, 1.0), Gaussian(0.5, 0.1),
              Gaussian(0.5, 0.1, minimum=0.0, maximum=1.0), Sine()]
    joint = JointPrior(priors, names=["a", "b", "c", "d"])
    rebuilt = joint_prior_from_specs(
        json.loads(json.dumps(joint.spec())))  # through json, as in HDF5
    assert rebuilt.names == joint.names
    assert np.allclose(rebuilt.mins, joint.mins)
    assert np.allclose(rebuilt.maxes, joint.maxes)
    pts = np.array([[0.3, 0.5, 0.9, 1.0], [0.7, 0.2, 0.1, 2.0]])
    assert np.allclose(rebuilt(pts), joint(pts))
    draws = rebuilt.sample(np.random.default_rng(1), size=100)
    assert np.all(draws >= rebuilt.mins) and np.all(draws <= rebuilt.maxes)


def test_callable_prior_not_reconstructable():
    joint = JointPrior(
        [CallablePrior(lambda x: 0.0 * x, 0.0, 1.0)], names=["a"])
    spec = joint.spec()
    assert spec[0]["type"] == "callable"
    with pytest.raises(ValueError, match="CallablePrior"):
        joint_prior_from_specs(spec)


def test_results_prior_from_file(tmp_path):
    run = make_run_dir(tmp_path, with_spec=True)
    res = convert(run)
    path = tmp_path / "results.h5"
    res.save(path)
    prior = Results.load(path).prior()
    assert prior.ndim == NDIM
    assert np.allclose(prior.mins, -5.0)
    # Spec-less run: clear error, arrays-only.
    res_nospec = convert(make_run_dir(tmp_path / "b", with_spec=False))
    with pytest.raises(ValueError, match="no prior spec"):
        res_nospec.prior()


def test_cli_smoke(tmp_path, monkeypatch, capsys):
    run = make_run_dir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["emridispatch-postprocess", str(run)])
    postprocess_main()
    out = run / "results.h5"
    assert out.exists()
    assert "wrote" in capsys.readouterr().out
    # Refuses overwrite without --force, allows with.
    with pytest.raises(SystemExit):
        postprocess_main()
    monkeypatch.setattr(
        sys, "argv", ["emridispatch-postprocess", str(run), "--force"])
    postprocess_main()


def test_rung_burn_guard(tmp_path):
    res = convert(make_run_dir(tmp_path))
    with pytest.raises(ValueError, match="reduce --burn"):
        res.posterior(burn=NSTEPS)
    assert res.posterior(burn=50).shape == (NSTEPS - 50, NDIM)
