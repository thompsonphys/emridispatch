"""Start a PE run without the installed CLI hooks.

Usage:
    python examples/run_impulse.py [path/to/config.yaml]

With no argument this script uses the impulse_config.yaml in this example folder. 
Requires the impulse extra: pip install -e .[impulse]
"""

# Keep per-eval BLAS/OMP single-threaded and neutralize lisatools' bare
# breakpoint(); must run before numpy is first imported to take effect.
from emridispatch.cli import set_env_guards

set_env_guards()

import os
import shutil
import sys

from emridispatch.config import load_config
from emridispatch.logging_utils import setup_logging
from emridispatch.pipeline import run_from_config


def main():
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "impulse_config.yaml")
    config_path = sys.argv[1] if len(sys.argv) > 1 else default

    cfg = load_config(config_path)

    # --- Run-parameter overrides -------------------------------------------
    # Anything in the YAML can be overridden on the loaded namespace before
    # run_from_config. Common ones:
    #
    # cfg.sampler.nsamples = 500
    # cfg.sampler.start_mode = "prior"
    # cfg.sampler.start_jitter = 0.1
    # cfg.run.seed = 42
    # cfg.run.outdir = "./my_run"
    # -----------------------------------------------------------------------

    # Same run-dir setup as the CLI: copy the config into the outdir
    # (so the run dir is self-contained for emridisp-postprocess) and log to
    # <outdir>/run.log.
    os.makedirs(cfg.run.outdir, exist_ok=True)
    dest = os.path.join(cfg.run.outdir, "config.yaml")
    if os.path.abspath(config_path) != os.path.abspath(dest):
        shutil.copyfile(config_path, dest)

    setup_logging(
        outdir=cfg.run.outdir,
        level=getattr(cfg.logging, "level", "INFO"),
        filename=getattr(cfg.logging, "file", "run.log"),
    )

    # resume=False ignores any existing checkpoint/cache in the outdir
    # (equivalent to the CLI's --no-resume).
    summary = run_from_config(cfg, resume=True)

    print(f"outputs written to {cfg.run.outdir}")
    if summary is not None:
        print(f"proposal acceptance: {summary.get('proposal_acceptance')}")
        print(f"swap acceptance:     {summary.get('swap_acceptance')}")


if __name__ == "__main__":
    main()
