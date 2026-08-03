"""Converter + HDF5 round-trip tests for the common results format."""

import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("h5py")

from conftest import (
    ERYN_IT, ERYN_NW, ERYN_NT, LNPRIOR, NSTEPS, TEMPS, TRUTH,
    make_eryn_run_dir, make_eryn_untempered_run_dir, make_run_dir)
from emridispatch.parameters import NDIM, PARAM_NAMES
from emridispatch.priors import (
    CallablePrior, Gaussian, JointPrior, Sine, Uniform, joint_prior_from_specs)
from emridispatch.results import _EXTRA_COLS, Results, convert, detect_backend
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
    # lnprob is tempered: lnprior + beta*lnlike, with lnprior = -1 everywhere
    # and betas [1.0, 0.0], so the beta=0 top rung keeps only the prior.
    assert np.allclose(res.lnprior, -1.0)
    assert np.allclose(res.lnprob[0], res.lnlike[0] - 1.0)
    assert np.allclose(res.lnprob[1], -1.0)
    # Temperatures from the last betas row; beta=0 -> inf.
    assert np.isclose(res.temperatures[0], 1.0)
    assert np.isinf(res.temperatures[1])
    # Per-walker acceptance fractions broadcast across steps.
    assert np.isclose(res.accepted[0].mean(), np.mean([1, 2, 3]) / ERYN_IT)
    assert np.isclose(res.accepted[1].mean(), np.mean([4, 5, 0]) / ERYN_IT)
    # Shared sidecars: truth + config metadata work as for impulse.
    assert np.allclose(res.truth_physical, TRUTH)
    assert res.config["backend"] == "eryn"


def test_convert_eryn_untempered_cold_rung_is_unit_temperature(tmp_path):
    res = convert(make_eryn_untempered_run_dir(tmp_path))
    assert res.samples.shape == (1, ERYN_IT * ERYN_NW, NDIM)
    # All-zero betas mean "eryn never wrote a ladder", not T = inf.
    assert res.temperatures.shape == (1,)
    assert np.isclose(res.temperatures[0], 1.0)
    assert np.isclose(res.accepted[0, 0], 2.0 / ERYN_IT)


def test_convert_eryn_keeps_genuine_infinite_top_rung(tmp_path):
    # A real Tmax=inf ladder has beta=0 only in the top rung, so the
    # untempered fallback must not fire.
    res = convert(make_eryn_run_dir(tmp_path))
    assert np.isclose(res.temperatures[0], 1.0)
    assert np.isinf(res.temperatures[ERYN_NT - 1])


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


@pytest.mark.parametrize("value", ["null", "''", "0", "[run, log]"])
def test_an_unusable_log_filename_attaches_no_log(tmp_path, value):
    """setup_logging writes no file for a falsy logging.file, so `file: null`
    is a valid config; reading the run dir back must honour it rather than
    joining it into a path."""
    run = make_run_dir(tmp_path)
    (run / "config.yaml").write_text(f"logging:\n  file: {value}\n")
    (run / "run.log").write_text("versions: ...\n")
    assert convert(run).run_log is None


def test_a_renamed_log_file_is_still_attached(tmp_path):
    """make_run_dir leaves a run.log behind, so the name has to come from the
    config rather than from whichever file happens to exist."""
    run = make_run_dir(tmp_path)
    (run / "config.yaml").write_text("logging:\n  file: sampler.log\n")
    (run / "run.log").write_text("the default name\n")
    (run / "sampler.log").write_text("the configured name\n")
    assert convert(run).run_log == "the configured name\n"


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


def test_step_counts_record_the_ensemble_width(tmp_path):
    er = convert(make_eryn_run_dir(tmp_path / "er"))
    assert er.nwalkers == ERYN_NW
    assert er.nsteps == ERYN_IT
    assert er.ndraws == ERYN_IT * ERYN_NW

    # Non-ensemble backends keep one row per step, so the two agree.
    imp = convert(make_run_dir(tmp_path / "imp"))
    assert imp.nwalkers == 1
    assert imp.nsteps == imp.ndraws == NSTEPS


def test_rung_burn_counts_steps_not_rows(tmp_path):
    res = convert(make_eryn_run_dir(tmp_path))
    kept = res.rung(0, burn=1)
    assert len(kept) == (ERYN_IT - 1) * ERYN_NW
    # Param 0 is step + 0.1*walker, so burning one step must land on step 1.
    assert np.isclose(kept[0, 0], 1.0)
    assert np.allclose(kept[:ERYN_NW, 0], 1.0 + 0.1 * np.arange(ERYN_NW))


def test_rung_thin_drops_whole_steps_not_walkers(tmp_path):
    res = convert(make_eryn_run_dir(tmp_path))
    kept = res.rung(0, thin=2)
    steps = np.arange(0, ERYN_IT, 2)
    assert len(kept) == len(steps) * ERYN_NW
    expected = (steps[:, None] + 0.1 * np.arange(ERYN_NW)[None, :]).ravel()
    assert np.allclose(kept[:, 0], expected)
    # Every walker survives; thinning must not silently subset the ensemble.
    assert len({round((v % 1) * 10) for v in kept[:, 0]}) == ERYN_NW


def test_rung_burn_guard_reports_steps(tmp_path):
    res = convert(make_eryn_run_dir(tmp_path))
    with pytest.raises(ValueError, match=f"run's {ERYN_IT} time step"):
        res.rung(0, burn=ERYN_IT)


def test_rung_guard_counts_steps_not_rows_when_thinning(tmp_path):
    """One surviving step is nwalkers rows, which cleared a row-count guard.

    Those rows are the same step of a coupled ensemble, so returning them as a
    posterior would hand corner.corner a set with no independent draws at all.
    """
    res = convert(make_eryn_run_dir(tmp_path))
    assert res.ndraws == ERYN_IT * ERYN_NW
    with pytest.raises(ValueError, match="reduce --burn or --thin"):
        res.rung(0, thin=1000)
    # Two steps is the smallest request that is not degenerate.
    assert len(res.rung(0, thin=ERYN_IT // 2)) >= 2 * ERYN_NW


def test_roundtrip_preserves_nwalkers(tmp_path):
    res = convert(make_eryn_run_dir(tmp_path))
    path = tmp_path / "results.h5"
    res.save(path)
    back = Results.load(path)
    assert back.nwalkers == ERYN_NW
    assert back.nsteps == ERYN_IT
    assert np.allclose(back.rung(0, burn=2), res.rung(0, burn=2))


def _set_chain_width(run, ncols):
    """Rewrite every chain file at a different column count, as a run whose
    sampling vector had a different length would have written it."""
    for path in sorted(run.glob("chain_*.txt")):
        cols = np.loadtxt(path, ndmin=2)
        keep = min(ncols, cols.shape[1])
        wide = np.zeros((len(cols), ncols))
        wide[:, :keep] = cols[:, :keep]
        np.savetxt(path, wide)


@pytest.mark.parametrize("extra", [-1, 1, 3])
def test_impulse_converter_rejects_a_wrong_chain_width(tmp_path, extra):
    """Reading a chain file is positional, so a width other than the expected
    one shifts every column after the parameters: one extra parameter column
    makes that parameter the lnlike, and the temperature read from `accepted`.
    Every value stays plausible, so only the width itself can catch it."""
    run = make_run_dir(tmp_path)
    ncols = NDIM + _EXTRA_COLS + extra
    _set_chain_width(run, ncols)
    with pytest.raises(ValueError, match=f"{ncols} column"):
        convert(run)


@pytest.mark.filterwarnings("ignore:.*input contained no data.*")
def test_an_empty_chain_file_is_reported_as_empty_not_as_a_bad_width(tmp_path):
    """A run killed before impulse's first save_freq flush leaves chain files
    with no rows, which np.loadtxt reports as shape (0, 1) -- a width the
    column check would otherwise blame on a mismatched version."""
    run = make_run_dir(tmp_path)
    (run / "chain_0.txt").write_text("")
    with pytest.raises(ValueError, match="empty chain files"):
        convert(run)


def test_impulse_converter_rejects_rungs_of_differing_widths(tmp_path):
    """np.stack fails on this anyway, but two frames later and without naming
    the widths."""
    run = make_run_dir(tmp_path)
    cols = np.loadtxt(run / "chain_1.txt", ndmin=2)
    np.savetxt(run / "chain_1.txt", cols[:, :-1])
    with pytest.raises(ValueError, match="column"):
        convert(run)


def test_impulse_lnprob_is_tempered_and_lnprior_recovered(tmp_path):
    """impulse writes lnprior + lnlike/temp; the schema keeps that convention.

    The fixture uses a known constant lnprior, so a rung whose temperature is
    not 1 pins down that the tempering term was handled rather than assumed
    away.
    """
    res = convert(make_run_dir(tmp_path))
    assert np.allclose(res.lnprior, LNPRIOR)
    for r, temp in enumerate(TEMPS):
        assert np.allclose(res.lnprob[r], LNPRIOR + res.lnlike[r] / temp)
    # Rung 1 is tempered, so the untempered posterior must differ from lnprob.
    assert not np.allclose(res.lnprob[1], res.lnprior[1] + res.lnlike[1])
    # Rung 0 is beta = 1, where the two definitions coincide.
    assert np.allclose(res.lnprob[0], res.lnprior[0] + res.lnlike[0])


def test_impulse_inf_rung_lnprior_passes_through(tmp_path):
    """build_ladder always ends at np.inf, so beta = 0 on the top rung always.

    At beta = 0 the tempered posterior is the prior, so lnprior is lnprob
    verbatim and no likelihood term is evaluated. Where a -inf likelihood makes
    impulse's own lnprob column NaN (it computes lnprior + lnlike/temp, the same
    0*-inf), nothing is recoverable and the NaN is inherited, not manufactured.
    """
    run = make_run_dir(tmp_path, ladder=[1.0, 25.0, np.inf], dead_row=3)
    res = convert(run)
    assert res.betas[-1, 2] == 0.0
    assert np.isneginf(res.lnlike[2, 3])
    assert np.array_equal(res.lnprior[2], res.lnprob[2], equal_nan=True)
    # Live rows keep the known constant prior on every rung, inf rung included.
    live = np.arange(NSTEPS) != 3
    for r in range(3):
        assert np.allclose(res.lnprior[r][live], LNPRIOR)


def test_impulse_unrecoverable_lnprior_is_counted_not_silent(tmp_path, caplog):
    """beta > 0 with lnlike = -inf gives lnprob = -inf, which carries no
    information about lnprior. NaN is the right value; silence is not."""
    with caplog.at_level("WARNING"):
        res = convert(make_run_dir(tmp_path, ladder=[1.0, 25.0], dead_row=3))
    assert np.isnan(res.lnprior[0, 3])
    assert "carries no information about lnprior" in caplog.text
    live = np.arange(NSTEPS) != 3
    assert np.allclose(res.lnprior[0][live], LNPRIOR)


def test_impulse_upstream_nan_lnprob_is_reported(tmp_path, caplog):
    """A NaN already in the lnprob column is upstream data loss, not a
    converter artefact, so it gets its own message."""
    with caplog.at_level("WARNING"):
        convert(make_run_dir(tmp_path, ladder=[1.0, np.inf], dead_row=3))
    assert "already NaN in the raw chain" in caplog.text


def test_eryn_beta_zero_rung_survives_infinite_likelihood(tmp_path):
    """Same 0*-inf hazard in the forward direction: at beta = 0 the tempered
    lnprob is exactly lnprior, so it must stay finite."""
    res = convert(make_eryn_run_dir(tmp_path, dead_row=2))
    assert res.betas[-1, 1] == 0.0
    dead = slice(2 * ERYN_NW, 3 * ERYN_NW)
    assert np.all(np.isneginf(res.lnlike[1, dead]))
    assert np.all(np.isfinite(res.lnprob[1]))
    assert np.allclose(res.lnprob[1], res.lnprior[1])
    # The cold rung has beta = 1, so -inf likelihood legitimately gives -inf.
    assert np.all(np.isneginf(res.lnprob[0, dead]))


def test_temperature_ladder_rejects_non_positive_entries(tmp_path):
    with pytest.raises(ValueError, match="must be > 0"):
        convert(make_run_dir(tmp_path, ladder=[1.0, 0.0]))


def test_negative_beta_is_not_folded_into_infinity(tmp_path):
    res = convert(make_eryn_run_dir(tmp_path))
    res.betas = np.full((ERYN_IT, ERYN_NT), -0.25)
    # A corrupt ladder must stay visible rather than reading as T = inf.
    assert np.allclose(res.rung_temperatures(1), -4.0)


def test_eryn_lnprob_uses_per_step_betas(tmp_path):
    """Under adaptive tempering one ladder vector is wrong for every step but
    the last, so the tempered lnprob must use the step's own beta."""
    run = make_eryn_run_dir(tmp_path, adapt_betas=True)
    res = convert(run)
    beta_draws = res.betas_per_draw()
    assert np.allclose(res.lnprob, res.lnprior + beta_draws * res.lnlike)
    # A single-vector approximation would disagree on the earlier steps.
    assert not np.allclose(res.lnprob, res.lnprior + res.betas[-1][:, None] * res.lnlike)


def test_betas_history_is_stored_per_step(tmp_path):
    res = convert(make_eryn_run_dir(tmp_path))
    assert res.betas.shape == (ERYN_IT, ERYN_NT)
    assert np.allclose(res.betas[:, 0], 1.0)
    assert np.allclose(res.betas[:, 1], 0.0)
    assert not res.ladder_adapted()


def test_untempered_run_gets_unit_betas(tmp_path):
    # The all-zero beta fallback must reach lnprob, not just the temperature
    # label: otherwise the cold chain's posterior loses its likelihood term.
    res = convert(make_eryn_untempered_run_dir(tmp_path))
    assert np.allclose(res.betas, 1.0)
    assert np.allclose(res.lnprob, res.lnprior + res.lnlike)
    assert not res.ladder_adapted()


def test_ladder_adapted_detected_and_warned(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        res = convert(make_eryn_run_dir(tmp_path, adapt_betas=True))
    assert res.ladder_adapted()
    assert "temperature ladder adapted" in caplog.text
    # temperatures is the final ladder; the history spans more than that.
    hist = res.rung_temperatures(1)
    assert np.isclose(hist[-1], res.temperatures[1])
    assert hist.min() < hist.max()


def test_impulse_ladder_adaptation_also_detected(tmp_path):
    drifting = [np.full(NSTEPS, t) for t in TEMPS]
    drifting[1] = np.linspace(2.0, 3.0, NSTEPS)
    res = convert(make_run_dir(tmp_path, temps=drifting))
    assert res.ladder_adapted()
    assert np.isclose(res.temperatures[1], 3.0)
    # lnprior recovery must still track the moving beta step by step.
    assert np.allclose(res.lnprior, LNPRIOR)


def test_betas_per_draw_matches_step_major_layout(tmp_path):
    res = convert(make_eryn_run_dir(tmp_path, adapt_betas=True))
    bd = res.betas_per_draw()
    assert bd.shape == (ERYN_NT, ERYN_IT * ERYN_NW)
    for step in range(ERYN_IT):
        block = bd[:, step * ERYN_NW:(step + 1) * ERYN_NW]
        assert np.allclose(block, res.betas[step][:, None])


def test_roundtrip_preserves_lnprior_and_betas(tmp_path):
    res = convert(make_eryn_run_dir(tmp_path, adapt_betas=True))
    path = tmp_path / "results.h5"
    res.save(path)
    back = Results.load(path)
    assert np.allclose(back.lnprior, res.lnprior)
    assert np.allclose(back.lnprob, res.lnprob)
    assert np.allclose(back.betas, res.betas)
    assert back.ladder_adapted()


def test_load_rejects_other_format_with_postprocess_hint(tmp_path):
    import h5py

    path = tmp_path / "old.h5"
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = 0
    with pytest.raises(ValueError, match="emridisp-postprocess"):
        Results.load(path)
