"""Sampler-agnostic problem description + backend protocol.

Stores the RAW physical-space likelihood/prior; reparam wrapping (for
whitened-space MCMC backends) happens only via wrapped(), not here.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable, Optional, Protocol

import numpy as np

from emridispatch.priors import JointPrior
from emridispatch.reparam import Reparam, ReparamCallable

__all__ = ["SamplingProblem", "SamplerBackend"]


@dataclass
class SamplingProblem:
    ndim: int
    param_names: list
    lnlike: Callable            # RAW physical-space likelihood: f(x_vec) -> float
    prior: JointPrior           # physical-space structure (bounds/periodic/sample)
    reparam: Reparam            # identity when reparam_mode == "off"
    reparam_mode: str
    x0: np.ndarray              # physical-space start point
    proposal_cov: Optional[np.ndarray]   # physical-space proposal covariance
    periodic: dict              # index -> period, physical space
    truth: Optional[np.ndarray]          # physical-space truth (None: blind run)
    outdir: str
    seed: int
    meta: dict = field(default_factory=dict)

    @property
    def whitened(self) -> bool:
        return self.reparam_mode in ("auto", "grid")

    def wrapped(self):
        """Sampling-space view for MCMC backends: reparam-wrapped callables plus
        the transformed start / proposal covariance / truth.

        jacobian=True adds the grid-mode change-of-variables term (no-op
        for linear "auto"); periodic is untouched since it's disjoint
        from the reparam's intrinsic-parameter block.
        """
        if self.whitened:
            lnlike = ReparamCallable(self.lnlike, self.reparam)
            lnprior = ReparamCallable(self.prior, self.reparam, jacobian=True)
            x0 = self.reparam.to_u(self.x0)
            cov = (self.reparam.transform_cov(self.proposal_cov)
                   if self.proposal_cov is not None else None)
            truth = self.reparam.to_u(self.truth) if self.truth is not None else None
        else:
            lnlike, lnprior = self.lnlike, self.prior
            x0 = np.asarray(self.x0, dtype=float)
            cov, truth = self.proposal_cov, self.truth
        return SimpleNamespace(lnlike=lnlike, lnprior=lnprior, x0=x0,
                               proposal_cov=cov, periodic=dict(self.periodic),
                               truth=truth)


class SamplerBackend(Protocol):
    name: str

    def run(self, problem: SamplingProblem, cfg, resume: bool = True) -> dict:
        """Build the concrete sampler, sample, write run outputs under
        problem.outdir, and return a summary dict."""
