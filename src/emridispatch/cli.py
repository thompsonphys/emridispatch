"""CLI entry point: `emridispatch <config.yaml> [--no-resume] [--log-level ...]`.
"""

import os


def set_env_guards():
    """Force single-threaded BLAS/OMP env vars and disable breakpoint().

    Must run before numpy import to take effect. Neutralizes lisatools'
    bare breakpoint() in EMRITDIWaveform's except block so a waveform
    failure can't drop a non-interactive run into pdb.
    """
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_v, "1")
    os.environ.setdefault("PYTHONBREAKPOINT", "0")


def main(argv=None):
    set_env_guards()

    import argparse

    p = argparse.ArgumentParser(
        prog="emridispatch", description="EMRI parameter-estimation run")
    p.add_argument("config", help="path to a YAML config")
    p.add_argument("--no-resume", action="store_true",
                   help="ignore any existing checkpoint/cache in the outdir")
    p.add_argument("--log-level", default=None,
                   help="override the config logging level (DEBUG, INFO, ...)")
    args = p.parse_args(argv)

    import shutil

    from emridispatch.config import load_config
    from emridispatch.logging_utils import setup_logging

    cfg = load_config(args.config)

    # Copy the config into the outdir (multichain's per-seed convention), so
    # the run dir is self-contained for emridispatch-postprocess.
    os.makedirs(cfg.run.outdir, exist_ok=True)
    dest = os.path.join(cfg.run.outdir, "config.yaml")
    if os.path.abspath(args.config) != os.path.abspath(dest):
        shutil.copyfile(args.config, dest)

    log_file = getattr(cfg.logging, "file", "run.log")
    setup_logging(
        outdir=cfg.run.outdir,
        level=args.log_level or getattr(cfg.logging, "level", "INFO"),
        filename=log_file,
    )

    from emridispatch.pipeline import run_from_config

    if run_from_config(cfg, resume=not args.no_resume) is None:
        stale = os.path.exists(os.path.join(cfg.run.outdir, "run_summary.json"))
        what = ("run_summary.json in the outdir is from an earlier run"
                if stale else "no run_summary.json written")
        where = (f" See {os.path.join(cfg.run.outdir, log_file)}"
                 if log_file else "")
        raise SystemExit(f"sampler failed; {what}.{where}")


if __name__ == "__main__":
    main()
