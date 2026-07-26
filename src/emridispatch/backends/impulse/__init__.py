"""impulse.PTSampler backend (optional: `pip install emridispatch[impulse]`).

Custom jump proposals follow `proposal(chain_stats) -> (q, qxy)`.
Output: chain_N.txt per temperature under outdir.
"""

import json
import logging
import os

import numpy as np

from emridispatch.backends.impulse.ladder import build_ladder
from emridispatch.backends.impulse.mode_jumps import build_mode_jumps

logger = logging.getLogger(__name__)


def _coerce(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return float(o)


class ImpulseBackend:
    name = "impulse"

    def run(self, problem, cfg, resume=True):
        """Build the PTSampler, sample, and write run_summary.json.

        Returns the summary dict, or None if sample() raised ValueError.
        """
        try:
            from impulse import PTSampler
        except ImportError as err:
            raise ImportError(
                "the impulse sampler is required for the 'impulse' backend. "
                "Install with `pip install emridispatch[impulse]`."
            ) from err

        imp = cfg.sampler.impulse
        ladder = build_ladder(imp.ladder)
        ntemps = len(ladder)

        # Ladder adaptation (Vousden+2016), OFF by default: it misallocates rungs
        # across the EMRI phase transition. Freeze = effectively-infinite timescale.
        adapt = bool(imp.ladder.adapt)
        adapt_t0 = float(imp.ladder.adapt_t0)
        adapt_nu = float(imp.ladder.adapt_nu) if adapt else 1e12

        method = imp.mode_jump.method
        mode_jump_weight = imp.mode_jump.weight
        nsamples = cfg.sampler.nsamples
        threads = imp.threads  # MUST be 1: FEW's generator is not thread-safe
        outdir = problem.outdir
        seed = problem.seed

        # Sampling-space view: reparam-wrapped callables, transformed start/cov.
        w = problem.wrapped()

        sampler = PTSampler(
            ndim=problem.ndim, lnlike=w.lnlike, lnprior=w.lnprior,
            sample_mean=w.x0.copy(), sample_cov=w.proposal_cov,
            ntemps=ntemps, ladder=ladder, swap_steps=1,
            inf_temp=False,  # inf chain already the top rung of ladder
            adapt_t0=adapt_t0, adapt_nu=adapt_nu,  # frozen when adapt is off
            de_weight=50.0, cov_update=imp.cov_update,
            save_freq=imp.save_freq,
            seed=seed, vectorized=False, threads=threads,
            periodic=w.periodic, outdir=outdir, resume=resume,
        )

        # Adaptive mode-jump proposal(s), learning modes online from a shared
        # cross-chain pool ("none" adds none). The GMM models the (whitened)
        # intrinsic block = the reparam indices.
        mode_jumps, mode_pool = build_mode_jumps(
            method, problem.ndim, dims=np.asarray(problem.reparam.idx),
            weight=mode_jump_weight, seed=seed)
        for jump, jw in mode_jumps:
            sampler.add_custom_jump(jump, weight=jw)

        logger.info("impulse: ntemps=%d nsamples=%d threads=%d outdir=%s",
                    ntemps, nsamples, threads, outdir)
        logger.info("ladder (dense low-T): %s",
                    np.array2string(ladder, precision=1, max_line_width=120))
        logger.info("ladder adapt: %s%s", adapt,
                    f" (t0={adapt_t0:g}, nu={adapt_nu:g})" if adapt else " (frozen)")
        logger.info("mode-jump method: %s (%d custom jump(s))",
                    method, len(mode_jumps))

        try:
            sampler.sample(w.x0.copy(), nsamples, thin=1)
        except ValueError as err:
            logger.error("sample() raised ValueError: %s", err)
            for i, cs in enumerate(sampler.multi_chain_stats.chain_stats):
                logger.error("chain %d: mean=%s cov=%s",
                             i, cs.sample_mean, cs.sample_cov)
            return None

        summary = {
            "config": dict(problem.meta,
                           backend=self.name, ntemps=ntemps,
                           mode_jump_method=method, nsamples=nsamples,
                           seed=seed),
            "proposal_acceptance": sampler.proposal_acceptance_rates(),
            "swap_acceptance": sampler.ptstate.compute_accept_ratio(),
        }
        summary_path = os.path.join(outdir, "run_summary.json")
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2, default=_coerce)
        logger.info("wrote run summary -> %s", summary_path)
        return summary
