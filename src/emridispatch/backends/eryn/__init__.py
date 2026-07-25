"""eryn EnsembleSampler backend (optional: `pip install emridispatch[eryn]`).

Everything eryn-specific lives here: the whitened-prior logpdf adapter, the
walker scatter around the SamplingProblem start point, the tempering_kwargs
mapping, the eryn_chain.h5 HDF backend + resume semantics, and the acceptance
reporting. The likelihood/prior/reparam come unchanged from problem.wrapped().

Config (`sampler:` section; the eryn subsection is all optional):
    backend: eryn
    nsamples: 10000         # stored steps per invocation (resume appends)
    eryn:
      nwalkers: 32          # ensemble walkers per temperature rung
      ntemps: 1             # 1 -> untempered; >1 builds an eryn ladder
      Tmax: null            # top ladder temperature (.inf allowed)
      adaptive_temps: true  # Vousden ladder adaptation
      adaptation_lag: 10000
      adaptation_time: 100
      stop_adaptation: -1   # step to freeze the ladder (-1: never)
      burn: 0               # in-sampler burn-in (not stored); skipped on resume
      thin_by: 1
      progress: false
      start_spread: 1.0     # walker scatter around the start, in proposal sigmas
      move: stretch         # the only supported move in this version
"""

import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

CHAIN_NAME = "eryn_chain.h5"


def _coerce(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return float(o)


class _WhitenedPrior:
    """problem.wrapped().lnprior -> the eryn prior interface.

    eryn only needs .logpdf (batched) plus a .key_order attribute in the
    non-reversible-jump path. key_order MUST be a scalar: h5py round-trips
    list attrs as numpy arrays and eryn's resume equality check on dicts of
    arrays raises ValueError; a scalar int compares cleanly.
    """

    def __init__(self, lnprior, ndim):
        self.lnprior = lnprior
        self.ndim = int(ndim)
        self.key_order = int(ndim)

    def logpdf(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        # Row loop: the grid-reparam jacobian is scalar-only, and the prior is
        # numpy-cheap next to the likelihood.
        out = np.array([float(self.lnprior(row)) for row in x])
        return np.where(np.isnan(out), -np.inf, out)  # eryn raises on NaN


def _initial_walkers(x0, proposal_cov, prior, ntemps, nwalkers, rng,
                     spread=1.0, max_tries=100):
    """(ntemps, nwalkers, ndim) scatter around the whitened start point.

    Draws x0 + spread * chol(proposal_cov) @ randn per walker, redrawing rows
    with non-finite prior; stubborn rows fall back to x0 exactly (x0 has
    finite prior by construction).
    """
    x0 = np.asarray(x0, dtype=float)
    ndim = x0.size
    if proposal_cov is not None:
        try:
            chol = np.linalg.cholesky(np.asarray(proposal_cov, dtype=float))
        except np.linalg.LinAlgError:
            chol = np.diag(np.sqrt(np.abs(np.diag(proposal_cov))))
    else:
        chol = np.zeros((ndim, ndim))

    coords = np.empty((ntemps * nwalkers, ndim))
    todo = np.ones(len(coords), dtype=bool)
    for _ in range(max_tries):
        n = int(todo.sum())
        if n == 0:
            break
        draws = x0 + spread * rng.standard_normal((n, ndim)) @ chol.T
        coords[todo] = draws
        todo[todo] = ~np.isfinite(prior.logpdf(draws))
    if todo.any():
        logger.warning("eryn: %d walker start(s) had no finite prior after %d "
                       "redraws; pinned to the start point", int(todo.sum()),
                       max_tries)
        coords[todo] = x0
    return coords.reshape(ntemps, nwalkers, ndim)


class ErynBackend:
    name = "eryn"

    def run(self, problem, cfg, resume=True):
        """Build the EnsembleSampler from the SamplingProblem, sample, and
        write run_summary.json. Returns the summary dict (None if run_mcmc
        raised a ValueError, mirroring the impulse backend contract)."""
        try:
            from eryn.backends import HDFBackend
            from eryn.ensemble import EnsembleSampler
            from eryn.state import State
        except ImportError as err:
            raise ImportError(
                "the eryn sampler is required for the 'eryn' backend. "
                "Install with `pip install emridispatch[eryn]`."
            ) from err

        # eryn <= 1.2.6 calls np.in1d, removed in numpy 2.0.
        if not hasattr(np, "in1d"):
            np.in1d = np.isin

        e = cfg.sampler.eryn
        nwalkers = int(e.nwalkers)
        ntemps = int(e.ntemps)
        nsamples = int(cfg.sampler.nsamples)
        Tmax = e.Tmax
        adaptive_temps = bool(e.adaptive_temps)
        adaptation_lag = int(e.adaptation_lag)
        adaptation_time = int(e.adaptation_time)
        stop_adaptation = int(e.stop_adaptation)
        burn = int(e.burn)
        thin_by = int(e.thin_by)
        progress = bool(e.progress)
        start_spread = float(e.start_spread)
        if nwalkers < 2 * problem.ndim:
            raise ValueError(
                f"sampler.eryn.nwalkers must be at least 2*ndim = "
                f"{2 * problem.ndim} (got {nwalkers}): eryn's red-blue stretch "
                "move splits the ensemble into two halves and each half must "
                "span the parameter space")

        move = str(e.move).lower()
        if move != "stretch":
            raise ValueError(
                f"sampler.eryn.move {move!r} not supported by the eryn backend: "
                "only 'stretch' is wired in (custom moves need eryn's "
                "temperature control and periodic containers handed in "
                "manually)")

        outdir = problem.outdir
        seed = problem.seed
        # eryn snapshots the global numpy RandomState at construction; a local
        # generator drives the walker scatter.
        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        # Sampling-space view: reparam-wrapped callables, transformed start/cov.
        w = problem.wrapped()
        prior = _WhitenedPrior(w.lnprior, problem.ndim)
        priors = {"model_0": prior}
        periodic = ({"model_0": {int(i): float(p)
                                 for i, p in w.periodic.items()}}
                    if w.periodic else None)

        tempering_kwargs = {}
        if ntemps > 1:
            tempering_kwargs = dict(
                ntemps=ntemps, adaptive=adaptive_temps,
                adaptation_lag=adaptation_lag,
                adaptation_time=adaptation_time,
                stop_adaptation=stop_adaptation)
            if Tmax is not None:
                tempering_kwargs["Tmax"] = float(Tmax)

        chain_path = os.path.join(outdir, CHAIN_NAME)
        if not resume and os.path.exists(chain_path):
            logger.info("eryn: fresh start requested; removing %s", chain_path)
            os.remove(chain_path)
        os.makedirs(outdir, exist_ok=True)
        backend = HDFBackend(chain_path)
        resuming = (resume and os.path.exists(chain_path)
                    and backend.initialized and backend.iteration > 0)

        sampler = EnsembleSampler(
            nwalkers, problem.ndim, w.lnlike, priors,
            tempering_kwargs=tempering_kwargs,
            moves=None,  # default StretchMove(a=2), tempering+periodic wired
            backend=backend, vectorize=False, periodic=periodic,
        )

        if resuming:
            logger.info("eryn: resuming from %s at iteration %d",
                        chain_path, int(backend.iteration))
            initial_state = None  # the sampler loaded the last sample + betas
        else:
            coords = _initial_walkers(w.x0, w.proposal_cov, prior,
                                      max(ntemps, 1), nwalkers, rng,
                                      spread=start_spread)
            initial_state = State({"model_0": coords})

        logger.info("eryn: ntemps=%d nwalkers=%d nsamples=%d thin_by=%d "
                    "outdir=%s", ntemps, nwalkers, nsamples, thin_by, outdir)

        try:
            sampler.run_mcmc(initial_state, nsamples,
                             burn=(burn if burn > 0 and not resuming else None),
                             thin_by=thin_by, progress=progress)
        except ValueError as err:
            logger.error("run_mcmc raised ValueError: %s", err)
            return None

        it = max(int(backend.iteration), 1)
        summary = {
            "config": dict(problem.meta,
                           backend=self.name, ntemps=ntemps,
                           nwalkers=nwalkers, nsamples=nsamples,
                           thin_by=thin_by, seed=seed),
            # Cumulative per-walker acceptance counts -> per-rung mean rates.
            "proposal_acceptance": (backend.accepted
                                    / (it * thin_by)).mean(axis=1),
            "swap_acceptance": backend.swaps_accepted / (it * thin_by * nwalkers),
        }
        summary_path = os.path.join(outdir, "run_summary.json")
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2, default=_coerce)
        logger.info("wrote run summary -> %s", summary_path)
        return summary
