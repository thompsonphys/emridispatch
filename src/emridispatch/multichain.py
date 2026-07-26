"""Launch N independent EMRI PT-MCMC cold chains for a cross-chain R-hat.

start_mode sets start dispersion: prior=blind, fisher=truth+jitter*sigma,
truth=all at injection (R-hat then optimistic). First seed builds and
caches the Fisher prior box; later seeds reuse it, skipping their own.
"""

from emridispatch.cli import set_env_guards

set_env_guards()  # entry-point script: guards before numpy-heavy imports

import argparse
import copy
import logging
import os
import shutil
import subprocess
import sys

import yaml

from emridispatch.config import load_config
from emridispatch.logging_utils import setup_logging
from emridispatch.results import DEFAULT_NAME, convert, is_complete

logger = logging.getLogger(__name__)


def outdir_for(mc_root, seed):
    return os.path.join(mc_root, f"seed_{seed}")


def _ensure_results(outdir):
    """Write results.h5 into a finished seed dir if missing."""
    path = os.path.join(outdir, DEFAULT_NAME)
    if os.path.exists(path):
        return
    try:
        convert(outdir).save(path)
        logger.info("wrote %s", path)
    except ImportError as err:
        logger.warning("skipping %s (%s); diagnostics will convert in memory",
                       path, err)


def _gpu_cleanup():
    """Release GPU/host memory between in-process chains so they don't accumulate."""
    import gc

    gc.collect()
    try:
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def _seed_config(cfg_base, seed, outdir, start_mode, start_jitter, nsamples):
    """Deep-copy the loaded config namespace and apply this seed's run overrides."""
    cfg = copy.deepcopy(cfg_base)
    cfg.run.seed = seed
    cfg.run.outdir = outdir
    cfg.sampler.start_mode = start_mode
    cfg.sampler.start_jitter = start_jitter
    if nsamples is not None:
        cfg.sampler.nsamples = nsamples
    return cfg


def _write_seed_yaml(base_raw, seed, outdir, start_mode, start_jitter, nsamples):
    """Derived YAML for the --isolate subprocess path."""
    r = copy.deepcopy(base_raw)
    r["run"].update(seed=seed, outdir=outdir)
    s = r.setdefault("sampler", {})
    s.update(start_mode=start_mode, start_jitter=start_jitter)
    if nsamples is not None:
        s["nsamples"] = nsamples
    path = os.path.join(outdir, "config.yaml")
    with open(path, "w") as fh:
        yaml.safe_dump(r, fh, sort_keys=False)
    return path


def run_seed(cfg_base, base_raw, seed, mc_root, ref_cache, start_mode,
             start_jitter, nsamples, isolate):
    outdir = outdir_for(mc_root, seed)
    # Fresh dir (no stale checkpoint), but seed the shared Fisher cache if we have one.
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)
    if ref_cache and os.path.exists(ref_cache):
        shutil.copy(ref_cache, os.path.join(outdir, "prior_bounds.npz"))

    logger.info("=== run seed=%s start=%s %s -> %s", seed, start_mode,
                "(isolated)" if isolate else "(in-process)", outdir)
    if isolate:
        cfg_path = _write_seed_yaml(base_raw, seed, outdir, start_mode, start_jitter, nsamples)
        rc = subprocess.run([sys.executable, "-m", "emridispatch.cli", cfg_path]).returncode
        ok = rc == 0 and is_complete(outdir)
        if not ok:
            logger.error("seed %s FAILED (rc=%s)", seed, rc)
    else:
        from emridispatch.pipeline import run_from_config

        cfg = _seed_config(cfg_base, seed, outdir, start_mode, start_jitter, nsamples)
        try:
            run_from_config(cfg)
        finally:
            _gpu_cleanup()
        ok = is_complete(outdir)
        if not ok:
            logger.error("seed %s FAILED (no run_summary.json)", seed)
    return ok


def main():
    p = argparse.ArgumentParser(description="Launch independent EMRI cold chains for R-hat")
    p.add_argument("base", help="base YAML config")
    p.add_argument("--outdir", default="./chains_mc",
                   help="root directory for the per-seed run dirs (default ./chains_mc)")
    p.add_argument("--nchains", type=int, default=4, help="number of independent chains (default 4)")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="explicit seeds (overrides --nchains); default base_seed + 0..nchains-1")
    p.add_argument("--start-mode", choices=["truth", "prior", "fisher"], default="fisher",
                   help="start dispersion (default fisher). 'truth' gives an optimistic R-hat.")
    p.add_argument("--jitter", type=float, default=None, help="fisher-mode dispersion in sigmas (default: base config)")
    p.add_argument("--nsamples", type=int, default=None, help="override sampler.nsamples for every chain")
    p.add_argument("--isolate", action="store_true", help="run each chain in a fresh subprocess (bulletproof GPU teardown)")
    p.add_argument("--resume", action="store_true", help="skip seeds whose dir already finished")
    p.add_argument("--analyze-only", action="store_true", help="skip running; just re-report finished dirs")
    p.add_argument("--no-report", action="store_true", help="skip the diagnostics report at the end")
    p.add_argument("--log-level", default=None, help="override the config logging level")
    args = p.parse_args()

    cfg_base = load_config(args.base)
    mc_root = os.path.abspath(args.outdir)
    os.makedirs(mc_root, exist_ok=True)
    setup_logging(outdir=mc_root,
                  level=args.log_level or getattr(cfg_base.logging, "level", "INFO"),
                  filename=getattr(cfg_base.logging, "file", "run.log"))

    base_seed = int(cfg_base.run.seed)
    start_jitter = args.jitter if args.jitter is not None else float(cfg_base.sampler.start_jitter)
    seeds = args.seeds if args.seeds is not None else [base_seed + i for i in range(args.nchains)]

    if args.start_mode == "truth":
        logger.warning("start-mode 'truth' starts every chain in the same basin -> "
                       "R-hat will be optimistic. Use 'fisher' or 'prior' for a "
                       "real convergence test.")

    # base_raw only needed for the --isolate (subprocess) path's derived YAML.
    base_raw = None
    if args.isolate:
        with open(args.base) as fh:
            base_raw = yaml.safe_load(fh)

    ref_cache = None
    done = []
    for seed in seeds:
        outdir = outdir_for(mc_root, seed)
        if not args.analyze_only:
            if args.resume and is_complete(outdir):
                logger.info("=== skip (done) seed=%s -> %s", seed, outdir)
            elif not run_seed(cfg_base, base_raw, seed, mc_root, ref_cache,
                              args.start_mode, start_jitter, args.nsamples,
                              args.isolate):
                continue
        # Seed the shared Fisher cache from the first finished seed.
        if ref_cache is None:
            cand = os.path.join(outdir, "prior_bounds.npz")
            if os.path.exists(cand):
                ref_cache = cand
        if is_complete(outdir):
            _ensure_results(outdir)
            done.append(outdir)

    logger.info("finished cold chains (%d/%d): %s", len(done), len(seeds), done)
    if not done:
        logger.warning("no completed chains -> nothing to diagnose")
        return
    if args.no_report:
        logger.info("re-run diagnostics with:\n  emridispatch-diagnostics %s --sweep",
                    " ".join(done))
        return

    from emridispatch.diagnostics import report
    report(done, burn=0)


if __name__ == "__main__":
    main()
