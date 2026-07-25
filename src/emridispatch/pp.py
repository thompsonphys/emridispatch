"""P-P (probability-probability) posterior-calibration test for the EMRI sampler.

N full PE runs, each on a different injection whose truth is drawn uniformly
from one fixed fiducial prior box; for a calibrated pipeline the rank of the
truth within each 1-D marginal posterior is Uniform(0,1) across runs.

Design (deliberate, for exact P-P validity):
  * The prior box is built once, at the fiducial injection (the config's
    injection: block), scaled by prior.box_scale and not re-centred on each run's
    truth. Truths are then genuine draws from the sampling prior, and the reparam
    whitening is anchored at the fiducial, so no truth information leaks into any
    run. Every run gets the reference cache copied in, which routes the sampler
    down its resume path (Fisher skipped, truth-centred box build bypassed).
  * SNR calibration is off for P-P runs (inj_snr -> null): rescaling distance
    to a fixed SNR would overwrite the drawn distance truth. Realized optimal
    SNR varies run to run and is recorded in each run_summary.json.
  * Noise on: each run gets its own PSD noise realization (noise_seed = base+i).
  * Chains start blind (start_mode 'prior').

Ranks are computed in physical coordinates (the auto-reparam whitening mixes
dimensions; per-parameter ranks in u-space would be meaningless).

Caveat: P-P assumes every posterior is sampled to convergence. With blind starts
and off-centre truths, sampler failures show up as rank pile-up at 0/1.

Each likelihood call is a full EMRI+TDI waveform -> production sweeps want a GPU
node. Runs are sequential; --isolate gives per-run subprocess GPU teardown.

Usage:
    emridispatch-pp my_config.yaml --nruns 20
    emridispatch-pp my_config.yaml --resume        # skip finished inj dirs
    emridispatch-pp my_config.yaml --analyze-only  # just re-tabulate + re-plot
"""

from emridispatch.cli import set_env_guards

set_env_guards()  # entry-point script: guards before numpy-heavy imports

import argparse
import copy
import json
import logging
import os
import shutil
import subprocess
import sys

import numpy as np
import yaml

from emridispatch.bounds import cache_path, build_prior_bounds, save_prior_bounds
from emridispatch.config import load_config
from emridispatch.diagnostics import load_cold_chain
from emridispatch.logging_utils import setup_logging
from emridispatch.parameters import NDIM, PARAM_NAMES, truth_vector
from emridispatch.results import is_complete

logger = logging.getLogger(__name__)

# Waveform-validity limits (FastKerrEccentricEquatorial domain); draws outside
# are rejected and redrawn. Keys index the 12-D sampling vector.
VALIDITY = {
    2: (-0.999, 0.999),   # a
    4: (0.0, 0.75),       # e
    5: (1e-3, np.inf),    # dist > 0
}
P_SEP_BUFFER = 0.2        # p must clear the separatrix-ish floor: p > 6 + 2e + buffer


def _valid_truth(vec):
    for idx, (lo, hi) in VALIDITY.items():
        if not (lo <= vec[idx] <= hi):
            return False
    e = vec[4]
    if vec[3] <= 6.0 + 2.0 * e + P_SEP_BUFFER:
        return False
    return True


def draw_truth(mins, maxes, rng, fiducial_inj, max_tries=1000):
    """One injection dict, truth drawn uniformly from the fiducial prior box.

    Masses are drawn in ln-space (the box's native rows 0/1) and exponentiated;
    x / phi_theta stay at the fiducial (not sampled -- equatorial model). Draws
    violating waveform validity are rejected and redrawn (this tilts the effective
    prior only where the likelihood would be -inf anyway).
    """
    for _ in range(max_tries):
        vec = rng.uniform(mins, maxes)
        if _valid_truth(vec):
            break
    else:
        raise RuntimeError("could not draw a valid truth inside the prior box")
    inj = dict(fiducial_inj)
    inj.update({
        "mass_1": float(np.exp(vec[0])), "mass_2": float(np.exp(vec[1])),
        "a": float(vec[2]), "p": float(vec[3]), "e": float(vec[4]),
        "luminosity_distance": float(vec[5]),
        "q_s": float(vec[6]), "phi_s": float(vec[7]),
        "q_k": float(vec[8]), "phi_k": float(vec[9]),
        "phi_phi": float(vec[10]), "phi_r": float(vec[11]),
    })
    return inj


def ensure_reference_cache(cfg, outroot):
    """Build the fiducial prior box once (Fisher at the config injection,
    SNR-calibrated so the box-centre distance is sane) and cache it under
    <outroot>/reference/. All P-P runs share this cache."""
    ref_dir = os.path.join(outroot, "reference")
    ref_cache = cache_path(ref_dir)
    if os.path.exists(ref_cache):
        logger.info("reference cache found: %s", ref_cache)
        return ref_cache

    logger.info("building reference prior box (one-off Fisher at the fiducial "
                "injection)...")
    from emridispatch.fisher import get_fisher_provider
    from emridispatch.pipeline import fisher_key_from_config
    from emridispatch.response import build_injection_model

    box_scale = float(cfg.prior.box_scale)
    model = build_injection_model(cfg)
    truth_vec = truth_vector(model.injection_parameters)
    provider = get_fisher_provider(cfg)
    fisher = provider.compute(
        model.injection_parameters,
        duration=cfg.data.duration, delta_t=cfg.data.delta_t,
        use_gpu=cfg.prior.fisher_use_gpu)
    mins, maxes, sample_cov, reparam = build_prior_bounds(
        fisher.sigmas, fisher.cov, fisher.order, model.injection_parameters,
        truth_vec, cfg.reparam.mode, np.array(cfg.reparam.idx),
        cfg.prior.angle_sigma, NDIM, box_scale=box_scale)
    save_prior_bounds(ref_cache, mins, maxes, sample_cov, reparam,
                      box_scale=box_scale, prec_dict=fisher.sigmas,
                      injection_parameters=model.injection_parameters,
                      reparam_mode=cfg.reparam.mode,
                      fisher_key=fisher_key_from_config(cfg))
    logger.info("reference cache built (box_scale=%g, provider=%s) -> %s",
                box_scale, provider.name, ref_cache)
    return ref_cache


def outdir_for(outroot, i):
    return os.path.join(outroot, f"inj_{i:03d}")


def _gpu_cleanup():
    import gc

    gc.collect()
    try:
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def _run_config(cfg_base, inj, i, outdir, nsamples):
    """Deep-copied config for P-P run i: drawn injection, blind start, free
    distance (no SNR calibration), per-run noise realization."""
    cfg = copy.deepcopy(cfg_base)
    cfg.injection = inj
    cfg.run.outdir = outdir
    cfg.run.seed = int(cfg_base.run.seed) + i
    cfg.sampler.start_mode = "prior"
    cfg.data.inj_snr = None
    cfg.data.add_noise = True
    cfg.data.noise_seed = int(getattr(cfg_base.data, "noise_seed", 0)) + i
    if nsamples is not None:
        cfg.sampler.nsamples = int(nsamples)
    return cfg


def _write_run_yaml(base_raw, cfg, outdir):
    """Derived YAML for the --isolate subprocess path (mirrors _run_config)."""
    r = copy.deepcopy(base_raw)
    r["injection"] = {k: float(v) for k, v in cfg.injection.items()}
    r["run"].update(seed=cfg.run.seed, outdir=outdir)
    r.setdefault("sampler", {}).update(start_mode="prior",
                                       nsamples=cfg.sampler.nsamples)
    r["data"].update(inj_snr=None, add_noise=True, noise_seed=cfg.data.noise_seed)
    path = os.path.join(outdir, "config.yaml")
    with open(path, "w") as fh:
        yaml.safe_dump(r, fh, sort_keys=False)
    return path


def run_injection(cfg_base, base_raw, inj, i, outdir, ref_cache, nsamples, isolate):
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)
    # Shared fiducial cache -> the sampler takes its resume path: Fisher skipped
    # and, crucially, the truth-centred box build bypassed (fixed-box P-P design).
    shutil.copy(ref_cache, cache_path(outdir))

    logger.info("=== pp run %d %s -> %s", i,
                "(isolated)" if isolate else "(in-process)", outdir)
    cfg = _run_config(cfg_base, inj, i, outdir, nsamples)
    if isolate:
        cfg_path = _write_run_yaml(base_raw, cfg, outdir)
        rc = subprocess.run([sys.executable, "-m", "emridispatch.cli", cfg_path]).returncode
        ok = rc == 0 and is_complete(outdir)
        if not ok:
            logger.error("pp run %d FAILED (rc=%s)", i, rc)
    else:
        from emridispatch.pipeline import run_from_config

        try:
            run_from_config(cfg)
        finally:
            _gpu_cleanup()
        ok = is_complete(outdir)
        if not ok:
            logger.error("pp run %d FAILED (no run_summary.json)", i)
    return ok


def compute_ranks(outdir, burn_frac):
    """Per-parameter rank of the truth in the 1-D marginals (physical coords)."""
    samples, _ = load_cold_chain(outdir, physical=True)
    burn = int(burn_frac * len(samples))
    post = samples[burn:]
    with open(os.path.join(outdir, "injection_truth.json")) as fh:
        truth = np.asarray(json.load(fh)["sampling_vector"], float)
    return {name: float(np.mean(post[:, j] < truth[j]))
            for j, name in enumerate(PARAM_NAMES)}


def analyze(outroot, run_dirs, burn_frac, make_plot=True):
    """Aggregate ranks over finished runs -> KS table, pp_ranks.json, pp_plot.png."""
    from scipy.stats import kstest

    ranks = {name: [] for name in PARAM_NAMES}
    snrs = []
    for d in run_dirs:
        r = compute_ranks(d, burn_frac)
        for name in PARAM_NAMES:
            ranks[name].append(r[name])
        with open(os.path.join(d, "run_summary.json")) as fh:
            snrs.append(float(json.load(fh)["config"].get("optimal_snr", np.nan)))

    nruns = len(run_dirs)
    print(f"\nP-P over {nruns} runs (burn_frac={burn_frac:g}); "
          f"realized SNR: min={np.nanmin(snrs):.1f} med={np.nanmedian(snrs):.1f} "
          f"max={np.nanmax(snrs):.1f}")
    print(f"{'param':>8s}  {'KS p':>8s}   ranks")
    results = {}
    for name in PARAM_NAMES:
        arr = np.sort(ranks[name])
        pval = float(kstest(arr, "uniform").pvalue)
        flag = "  <-- suspicious" if pval < 0.01 else ""
        print(f"{name:>8s}  {pval:8.3f}   {np.round(arr, 2)}{flag}")
        results[name] = {"ranks": ranks[name], "ks_pvalue": pval}

    out = {"nruns": nruns, "burn_frac": burn_frac, "run_dirs": run_dirs,
           "optimal_snrs": snrs, "params": results}
    ranks_path = os.path.join(outroot, "pp_ranks.json")
    with open(ranks_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {ranks_path}")

    if make_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        x = np.linspace(0, 1, 200)
        # Pointwise ~2/3-sigma binomial band around the diagonal.
        band = np.sqrt(x * (1 - x) / max(nruns, 1))
        ax.fill_between(x, x - band, x + band, color="0.85", label="1-sigma binomial")
        ax.fill_between(x, x - 3 * band, x + 3 * band, color="0.93", zorder=0)
        for name in PARAM_NAMES:
            arr = np.sort(ranks[name])
            ax.plot(np.append(arr, 1.0), np.append(np.arange(1, nruns + 1) / nruns, 1.0),
                    drawstyle="steps-post", lw=1, label=name)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("rank of truth in 1-D marginal")
        ax.set_ylabel("empirical CDF over runs")
        ax.set_title(f"P-P test ({nruns} runs)")
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        plot_path = os.path.join(outroot, "pp_plot.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"wrote {plot_path}")
    return out


def main():
    p = argparse.ArgumentParser(description="EMRI P-P posterior-calibration test")
    p.add_argument("base", help="base YAML config")
    p.add_argument("--nruns", type=int, default=None, help="override pp.nruns")
    p.add_argument("--nsamples", type=int, default=None, help="override pp.nsamples")
    p.add_argument("--isolate", action="store_true", help="run each injection in a fresh subprocess")
    p.add_argument("--resume", action="store_true", help="skip injection dirs that already finished")
    p.add_argument("--analyze-only", action="store_true", help="skip running; just re-tabulate + plot")
    p.add_argument("--no-plot", action="store_true", help="skip pp_plot.png")
    p.add_argument("--log-level", default=None, help="override the config logging level")
    args = p.parse_args()

    cfg = load_config(args.base)
    nruns = args.nruns if args.nruns is not None else int(getattr(cfg.pp, "nruns", 50))
    nsamples = args.nsamples if args.nsamples is not None else getattr(cfg.pp, "nsamples", None)
    outroot = os.path.abspath(getattr(cfg.pp, "outroot", "./chains_pp"))
    draw_seed = int(getattr(cfg.pp, "draw_seed", 7777))
    burn_frac = float(getattr(cfg.pp, "burn_frac", 0.25))
    os.makedirs(outroot, exist_ok=True)
    setup_logging(outdir=outroot,
                  level=args.log_level or getattr(cfg.logging, "level", "INFO"),
                  filename=getattr(cfg.logging, "file", "run.log"))

    base_raw = None
    if args.isolate:
        with open(args.base) as fh:
            base_raw = yaml.safe_load(fh)

    done = []
    if args.analyze_only:
        done = [outdir_for(outroot, i) for i in range(nruns)
                if is_complete(outdir_for(outroot, i))]
    else:
        ref_cache = ensure_reference_cache(cfg, outroot)
        box = np.load(ref_cache)
        mins, maxes = box["mins"], box["maxes"]

        for i in range(nruns):
            outdir = outdir_for(outroot, i)
            if args.resume and is_complete(outdir):
                logger.info("=== skip (done) pp run %d -> %s", i, outdir)
                done.append(outdir)
                continue
            # Draw seeded per run index -> the truth list is reproducible and
            # independent of which runs are skipped/resumed.
            rng = np.random.default_rng(draw_seed + i)
            inj = draw_truth(mins, maxes, rng, cfg.injection)
            if run_injection(cfg, base_raw, inj, i, outdir, ref_cache, nsamples, args.isolate):
                done.append(outdir)

    logger.info("finished pp runs: %d/%d", len(done), nruns)
    if not done:
        logger.warning("nothing to analyze")
        return
    analyze(outroot, done, burn_frac, make_plot=not args.no_plot)


if __name__ == "__main__":
    main()
