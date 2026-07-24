"""Logging setup for the emridispatch pipeline.

Stdlib logging only. setup_logging() is called from the CLI entry points (never
at import time) and configures the "emridispatch" package logger with a console
handler plus an optional per-run file handler in the output directory. Library
modules just do `logger = logging.getLogger(__name__)`.
"""

import logging
import os

PACKAGE_LOGGER = "emridispatch"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(outdir=None, level="INFO", filename="run.log"):
    """Configure the emridispatch package logger.

    Parameters
    ----------
    outdir : str or None
        Run output directory. When given (and filename is not None), a file
        handler writes to <outdir>/<filename> (directory created if needed).
    level : str or int
        Log level for the package logger ("DEBUG", "INFO", ...).
    filename : str or None
        Log file name relative to outdir; None disables the file handler.

    Idempotent: repeat calls replace the previous handlers rather than
    stacking duplicates (multichain / P-P drivers call once per run dir).
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
