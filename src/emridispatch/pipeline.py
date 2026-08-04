"""Backend-agnostic EMRI PE pipeline: config -> SamplingProblem -> backend.

12-D sampled vector: [ln m1, ln m2, a, p, e, dist, q_s, phi_s, q_k, phi_k,
phi_phi, phi_r]. resume=True restarts from the OUTDIR checkpoint/cache.
"""

import json
import logging
import os

import numpy as np

from emridispatch.backends import get_backend
from emridispatch.backends.base import SamplingProblem
from emridispatch.bounds import (
    build_prior_bounds, cache_path, fisher_cache_key, load_prior_bounds,
    save_prior_bounds)
from emridispatch.fisher import get_fisher_provider
from emridispatch.parameters import NDIM, PARAM_NAMES, truth_vector
from emridispatch.priors import joint_prior_from_config
from emridispatch.response import build_injection_model
from emridispatch.starts import initial_point

logger = logging.getLogger(__name__)


# Packages behind each sampler backend / response (TDI) model / Fisher
# provider, logged at run start for reproducibility (the log is archived into
# results.h5). Entries may be import names or distribution names.
_BACKEND_DISTS = {"impulse": ["impulse-mcmc"], "eryn": ["eryn"]}
_RESPONSE_DISTS = {
    "lisatools": ["lisaanalysistools", "fastemriwaveforms", "fastlisaresponse"],
    "toy": [],
}
_FISHER_DISTS = {
    "sef": ["stableemrifisher"],
    "auto": ["stableemrifisher"],   # auto uses SEF when importable
    "manual": [],
    "none": [],
}


def _dist_version(name):
    """Version for a distribution OR import name.

    Falls back to packages_distributions() when metadata.version() doesn't
    recognize name as a distribution (e.g. import `impulse` -> dist
    `impulse-mcmc`).
    """
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        pass
    dists = metadata.packages_distributions().get(name)
    if dists:
        return ", ".join(f"{d} {metadata.version(d)}"
                         for d in sorted(set(dists)))
    return "not installed"


def log_versions(cfg):
    """Log emridispatch + sampler-backend + response/TDI + Fisher versions."""

    def fmt(dists):
        return ", ".join(
            f"{d}={_dist_version(d)}" for d in dists) or "no external packages"

    logger.info("versions: emridispatch=%s", _dist_version("emridispatch"))
    backend = str(cfg.sampler.backend)
    logger.info("versions: sampler backend %r: %s",
                backend, fmt(_BACKEND_DISTS.get(backend, [backend])))
    response = str(getattr(cfg.data, "response", "lisatools"))
    logger.info("versions: response %r: %s",
                response, fmt(_RESPONSE_DISTS.get(response, [response])))
    fisher = str(cfg.prior.fisher).lower()
    logger.info("versions: fisher %r: %s",
                fisher, fmt(_FISHER_DISTS.get(fisher, [fisher])))


def _coerce(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return float(o)


def _write_injection_truth(outdir, inj_params, truth_vec):
    """Persist the injected truth for downstream plotting.

    Writes injection_truth.json: the 12-D sampling vector (masses in log,
    distance after SNR calibration) plus the full physical injection dict.
    """
    os.makedirs(outdir, exist_ok=True)
    truth = {
        "param_names": PARAM_NAMES,
        "sampling_vector": [float(v) for v in truth_vec],
        "sampling_truth": {n: float(v) for n, v in zip(PARAM_NAMES, truth_vec)},
        "injection": {k: _coerce(v) for k, v in dict(inj_params).items()},
    }
    with open(os.path.join(outdir, "injection_truth.json"), "w") as fh:
        json.dump(truth, fh, indent=2)


def fisher_key_from_config(cfg):
    channels = getattr(cfg.data, "channels", None)
    return fisher_cache_key(
        str(getattr(cfg.data, "tdi", "2nd generation")),
        bool(getattr(cfg.data, "foreground", True)),
        cfg.data.duration, cfg.data.delta_t,
        None if channels is None else list(channels),
    ) + f"|provider={get_fisher_provider(cfg).name}"


def build_problem(cfg, resume=True):
    """Construct a backend-agnostic SamplingProblem from a loaded config.

    Loads prior bounds from the outdir cache when present, else computes
    via the configured Fisher provider and caches them. No sampler import.
    """
    ndim = NDIM
    outdir = cfg.run.outdir
    seed = cfg.run.seed
    angle_sigma = cfg.prior.angle_sigma
    start_mode = str(cfg.sampler.start_mode).lower()
    start_jitter = float(cfg.sampler.start_jitter)

    periodic = {i: 2 * np.pi for i in cfg.prior.periodic_2pi_indices}
    reparam_mode = cfg.reparam.mode
    reparam_idx = np.array(cfg.reparam.idx)
    box_scale = float(cfg.prior.box_scale)

    model = build_injection_model(cfg)
    logger.info("injection SNR = %.4f (dist=%.4f)", model.optimal_snr,
                float(model.injection_parameters["luminosity_distance"]))

    truth_vec = truth_vector(model.injection_parameters)
    _write_injection_truth(outdir, model.injection_parameters, truth_vec)

    # Prior-bounds cache: compute the (possibly expensive) Fisher box once,
    # reuse thereafter.
    bounds_cache = cache_path(outdir)
    fisher_key = fisher_key_from_config(cfg)
    if resume and os.path.exists(bounds_cache):
        mins, maxes, sample_cov, reparam, reparam_mode = load_prior_bounds(
            bounds_cache, ndim, reparam_idx, reparam_mode, box_scale=box_scale,
            fisher_key=fisher_key)
        logger.info("resuming: loaded cached prior bounds + proposal from %s "
                    "(Fisher skipped)", bounds_cache)
    else:
        provider = get_fisher_provider(cfg)
        logger.info("fisher: computing via %r provider", provider.name)
        fisher = provider.compute(
            model.injection_parameters,
            duration=cfg.data.duration, delta_t=cfg.data.delta_t,
            use_gpu=cfg.prior.fisher_use_gpu)
        mins, maxes, sample_cov, reparam = build_prior_bounds(
            fisher.sigmas, fisher.cov, fisher.order,
            model.injection_parameters, truth_vec,
            reparam_mode, reparam_idx, angle_sigma, ndim, box_scale=box_scale)
        save_prior_bounds(bounds_cache, mins, maxes, sample_cov, reparam,
                          box_scale=box_scale, prec_dict=fisher.sigmas,
                          injection_parameters=model.injection_parameters,
                          reparam_mode=reparam_mode, fisher_key=fisher_key)
        logger.info("first run: computed prior bounds + proposal via %r "
                    "(box_scale=%g), cached to %s",
                    provider.name, box_scale, bounds_cache)

    # Structured joint prior: Uniform/PeriodicUniform box by default, then
    # per-parameter overrides from the config `priors:` section.
    prior = joint_prior_from_config(cfg, mins, maxes)
    if cfg.priors:
        logger.info("priors: overrides applied for %s", sorted(cfg.priors))

    # Persist the final per-parameter prior specs so postprocessing can
    # reconstruct the JointPrior (prior draws, reweighting) without the config.
    try:
        prior_spec = prior.spec()
    except TypeError as exc:
        logger.warning("prior spec not serialized: %s", exc)
    else:
        with open(os.path.join(outdir, "prior_spec.json"), "w") as fh:
            json.dump(prior_spec, fh, indent=2)

    reparam.save(os.path.join(outdir, "reparam_transform.npz"), reparam_mode)
    logger.info("reparam: mode=%s, block dims=%s", reparam_mode, list(reparam.idx))

    # Dispersed start in physical space, clipped to the (possibly overridden)
    # prior bounds; backends map it through the reparam via problem.wrapped().
    x0 = initial_point(start_mode, truth_vec, sample_cov, prior.mins, prior.maxes,
                       seed, start_jitter)
    logger.info("start mode: %s%s", start_mode,
                f" (jitter={start_jitter:g} sigma)" if start_mode == "fisher" else "")

    meta = dict(reparam_mode=reparam_mode, box_scale=box_scale,
                optimal_snr=model.optimal_snr,
                response=str(getattr(cfg.data, "response", "lisatools")),
                add_noise=bool(getattr(cfg.data, "add_noise", False)),
                noise_seed=int(getattr(cfg.data, "noise_seed", 0)))

    return SamplingProblem(
        ndim=ndim, param_names=list(PARAM_NAMES),
        lnlike=model, prior=prior,
        reparam=reparam, reparam_mode=reparam_mode,
        x0=x0, proposal_cov=sample_cov, periodic=periodic,
        truth=truth_vec, outdir=outdir, seed=seed, meta=meta,
    )


def run_from_config(cfg, resume=True):
    """Build the problem and hand it to the configured sampler backend.

    Returns the backend's summary dict (or None on a handled sampler failure).
    """
    backend_name = str(cfg.sampler.backend)
    backend = get_backend(backend_name)
    log_versions(cfg)
    problem = build_problem(cfg, resume=resume)
    logger.info("backend: %s", backend_name)
    return backend.run(problem, cfg, resume=resume)
