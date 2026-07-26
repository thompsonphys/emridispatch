"""Fisher-matrix providers: how the prior box and proposal covariance get sized.

Optional infrastructure; providers are sef (StableEMRIFisher), manual
(prior.sigmas / prior.covariance_file from config), or heuristic (rough
fallback). Selected via prior.fisher: auto | sef | manual | none, where
auto resolves to manual if configured, else sef if importable, else
heuristic. Providers implement

    compute(injection_parameters, duration=..., delta_t=...,
            use_gpu=None) -> FisherResult

carrying diagonal 1-sigma errors and the 6x6 intrinsic covariance in
linear coords, ordered [mass_1, mass_2, a, p, e, luminosity_distance].
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Protocol

import numpy as np

from emridispatch.parameters import INTRINSIC_ORDER

__all__ = ["FisherResult", "FisherProvider", "get_fisher_provider"]


@dataclass
class FisherResult:
    """Fisher products in linear coordinates, ordered as INTRINSIC_ORDER."""

    sigmas: dict          # injection-parameter name -> 1-sigma error
    cov: np.ndarray       # (6, 6) covariance, linear coords
    order: list           # ["mass_1", "mass_2", "a", "p", "e", "luminosity_distance"]

    def __post_init__(self):
        self.cov = np.asarray(self.cov, dtype=float)
        if list(self.order) != INTRINSIC_ORDER:
            raise ValueError(
                f"FisherResult order must be {INTRINSIC_ORDER}, got {self.order}")
        missing = [p for p in INTRINSIC_ORDER if p not in self.sigmas]
        if missing:
            raise ValueError(f"FisherResult sigmas missing {missing}")


class FisherProvider(Protocol):
    name: str

    def compute(self, injection_parameters, *, duration, delta_t,
                use_gpu=None) -> FisherResult: ...


def _sef_importable():
    try:
        import stableemrifisher  # noqa: F401

        return True
    except ImportError:
        return False


def get_fisher_provider(cfg):
    """Resolve the provider from cfg.prior.fisher (see module docstring)."""
    from emridispatch.fisher import manual as _manual

    kind = str(getattr(cfg.prior, "fisher", "auto")).lower()
    has_manual_cfg = (getattr(cfg.prior, "sigmas", None) is not None
                      or getattr(cfg.prior, "covariance_file", None) is not None)

    if kind == "auto":
        if has_manual_cfg:
            kind = "manual"
        elif _sef_importable():
            kind = "sef"
        else:
            kind = "none"

    if kind == "sef":
        from emridispatch.fisher.sef import SEFFisherProvider

        data = getattr(cfg, "data", SimpleNamespace())
        _channels = getattr(data, "channels", None)
        return SEFFisherProvider(
            tdi=str(getattr(data, "tdi", "2nd generation")),
            foreground=bool(getattr(data, "foreground", True)),
            channels=None if _channels is None else list(_channels))
    if kind == "manual":
        return _manual.ManualFisherProvider.from_config(cfg)
    if kind in ("none", "heuristic"):
        return _manual.HeuristicFisherProvider()
    raise ValueError(
        f"unknown prior.fisher setting {kind!r} "
        "(auto | sef | manual | none)")
