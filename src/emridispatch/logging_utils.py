"""Logging setup for the emridispatch pipeline.

setup_logging() must be called explicitly (never at import time) to
configure the "emridispatch" logger with a console + optional per-run
file handler; library modules just use logging.getLogger(__name__).
"""

import logging
import os

PACKAGE_LOGGER = "emridispatch"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(outdir=None, level="INFO", filename="run.log"):
    """Configure the emridispatch package logger.

    Idempotent: replaces prior handlers rather than stacking. Adds a file
    handler at <outdir>/<filename> only if both are given (dir auto-created).
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)
    logger.propagate = False

    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()

    fmt = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if outdir is not None and filename:
        os.makedirs(outdir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(outdir, filename))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def get_logger(name):
    """Module logger under the emridispatch namespace."""
    return logging.getLogger(name)
