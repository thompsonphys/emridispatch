"""Fisher-matrix providers: how the prior box and proposal covariance get sized.

The Fisher is OPTIONAL infrastructure. StableEMRIFisher (extras
`emridispatch[fisher]`) is one provider; a manual provider reads user-supplied
sigmas / a covariance file from the config; a heuristic provider produces rough
defaults so the pipeline runs end-to-end with no Fisher code installed at all.

Every provider implements

    provider.compute(injection_parameters, duration=..., delta_t=...,
                     use_gpu=None) -> FisherResult

with FisherResult carrying the diagonal 1-sigma errors, the full 6x6 intrinsic
covariance (linear coordinates), and the parameter order
[mass_1, mass_2, a, p, e, luminosity_distance].

Config (`prior:` section):
    fisher: auto | sef | manual | none
      auto   - manual if sigmas/covariance configured, else StableEMRIFisher if
               importable, else the heuristic fallback (with a loud warning).
      sef    - StableEMRIFisher, error if not installed.
      manual - prior.sigmas mapping and/or prior.covariance_file (npz).
      none   - heuristic defaults.
"""

from dataclasses import dataclass
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

        return SEFFisherProvider()
    if kind == "manual":
        return _manual.ManualFisherProvider.from_config(cfg)
    if kind in ("none", "heuristic"):
        return _manual.HeuristicFisherProvider()
    raise ValueError(
        f"unknown prior.fisher setting {kind!r} "
        "(auto | sef | manual | none)")
