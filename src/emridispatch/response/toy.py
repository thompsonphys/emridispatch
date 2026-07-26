"""Toy Gaussian injection model: no waveforms, no heavy dependencies.

ln L: independent Gaussian centred on truth in 12-D sampling coords.
Select with data.response: toy. NOT physics -- structure checks only.
"""

import logging

import numpy as np

from emridispatch.parameters import NDIM, truth_vector
from emridispatch.response import InjectionModel

logger = logging.getLogger(__name__)


class ToyGaussianLikelihood(InjectionModel):
    def __init__(self, injection_parameters, sigma_scale=0.05, snr=30.0):
        self.injection_parameters = dict(injection_parameters)
        self.truth = truth_vector(self.injection_parameters)
        # Per-dimension widths: relative to the truth scale, floored so zero
        # truths (e.g. a = 0) still get a finite width.
        self.sigma = sigma_scale * np.maximum(np.abs(self.truth), 1.0)
        self.optimal_snr = float(snr)
        logger.info("toy model: Gaussian likelihood, sigma_scale=%g", sigma_scale)

    @classmethod
    def from_config(cls, cfg):
        return cls(
            dict(cfg.injection),
            sigma_scale=float(getattr(cfg.data, "toy_sigma_scale", 0.05)),
            snr=float(getattr(cfg.data, "inj_snr", None) or 0.0),
        )

    def evaluate_likelihood(self, template_params) -> float:
        vec = truth_vector(template_params)
        return self(vec)

    def __call__(self, params) -> float:
        params = np.asarray(params, dtype=float)
        if params.shape != (NDIM,) or not np.all(np.isfinite(params)):
            return -np.inf
        z = (params - self.truth) / self.sigma
        return float(-0.5 * np.dot(z, z))
