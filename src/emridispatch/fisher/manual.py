"""Fisher providers that need no Fisher code: user-configured or heuristic.

ManualFisherProvider reads prior.sigmas (dict) and/or prior.covariance_file
(npz with `cov`[, `order`]); missing one is derived from the other, with
diagonal covariance when only sigmas are given. The box is
truth +/- prior.box_scale * sigma. HeuristicFisherProvider's widths are
NOT science-quality -- the box and proposal will be badly scaled.
"""

import logging

import numpy as np

from emridispatch.fisher import FisherResult
from emridispatch.parameters import INTRINSIC_ORDER

logger = logging.getLogger(__name__)


class ManualFisherProvider:
    name = "manual"

    def __init__(self, sigmas=None, cov=None):
        if sigmas is None and cov is None:
            raise ValueError(
                "prior.fisher: manual needs prior.sigmas and/or "
                "prior.covariance_file in the config")
        if cov is not None:
            cov = np.asarray(cov, dtype=float)
            if cov.shape != (6, 6):
                raise ValueError(f"manual covariance must be 6x6, got {cov.shape}")
        if sigmas is not None:
            missing = [p for p in INTRINSIC_ORDER if p not in sigmas]
            if missing:
                raise ValueError(f"prior.sigmas missing {missing}")
            sigmas = {p: float(sigmas[p]) for p in INTRINSIC_ORDER}
        self._sigmas = sigmas
        self._cov = cov

    @classmethod
    def from_config(cls, cfg):
        sigmas = getattr(cfg.prior, "sigmas", None)
        cov = None
        cov_file = getattr(cfg.prior, "covariance_file", None)
        if cov_file is not None:
            with np.load(cov_file) as d:
                cov = np.asarray(d["cov"], dtype=float)
                if "order" in d.files and list(d["order"]) != INTRINSIC_ORDER:
                    raise ValueError(
                        f"covariance_file order {list(d['order'])} != "
                        f"{INTRINSIC_ORDER}")
        return cls(sigmas=sigmas, cov=cov)

    def compute(self, injection_parameters, *, duration, delta_t, use_gpu=None):
        cov = self._cov
        sigmas = self._sigmas
        if cov is None:
            cov = np.diag([sigmas[p] ** 2 for p in INTRINSIC_ORDER])
        if sigmas is None:
            sigmas = {p: float(np.sqrt(cov[i, i]))
                      for i, p in enumerate(INTRINSIC_ORDER)}
        logger.info("fisher: manual provider (%s)",
                    "full covariance" if self._cov is not None else "diagonal sigmas")
        return FisherResult(sigmas=sigmas, cov=cov, order=list(INTRINSIC_ORDER))


class HeuristicFisherProvider:
    """Rough relative widths; enough to run the pipeline, NOT for science."""

    name = "heuristic"

    # Relative (mass_1, mass_2, luminosity_distance) and absolute (a, p, e)
    # 1-sigma scales -- order-of-magnitude EMRI-typical placeholders only.
    REL = {"mass_1": 1e-6, "mass_2": 1e-6, "luminosity_distance": 0.1}
    ABS = {"a": 1e-5, "p": 1e-5, "e": 1e-5}

    def compute(self, injection_parameters, *, duration, delta_t, use_gpu=None):
        logger.warning(
            "fisher: no Fisher provider available -> using HEURISTIC prior "
            "widths (relative %s, absolute %s). These are placeholders: the "
            "prior box and proposal covariance are NOT calibrated to the data. "
            "Install `emridispatch[fisher]` or configure prior.fisher: manual for "
            "real runs.", self.REL, self.ABS)
        sigmas = {}
        for p in INTRINSIC_ORDER:
            if p in self.REL:
                sigmas[p] = abs(float(injection_parameters[p])) * self.REL[p]
                if sigmas[p] == 0.0:
                    sigmas[p] = self.ABS.get(p, 1e-5)
            else:
                sigmas[p] = self.ABS[p]
        cov = np.diag([sigmas[p] ** 2 for p in INTRINSIC_ORDER])
        return FisherResult(sigmas=sigmas, cov=cov, order=list(INTRINSIC_ORDER))
