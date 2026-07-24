"""Prior box + proposal covariance + reparam whitening, with an OUTDIR cache.

The Fisher precision that sizes the prior is expensive, so the first run builds
the box + correlated proposal + reparam from it and caches them to
OUTDIR/prior_bounds.npz; every later restart loads the cache and skips the Fisher.

The Fisher compute lives behind emridispatch.fisher (optional providers).
Everything here is pure numpy given a FisherResult.

Sampling-vector order (masses in log), matching the likelihood:
    [ln m1, ln m2, a, p, e, dist, q_s, phi_s, q_k, phi_k, phi_phi, phi_r]
The first six are Fisher +/- SCALE*sigma boxes; the last six get their full
physical ranges (polar angles [0, pi]; azimuthal angles / phases [0, 2*pi]).
Periodicity is carried as metadata on the JointPrior; the box itself is a
plain bound.
"""

import logging
import os

import numpy as np

from emridispatch.parameters import ANGLE_RANGES, INTRINSIC_ORDER, LOG_PARAMS
from emridispatch.reparam import Reparam, GridReparam

logger = logging.getLogger(__name__)

CACHE_NAME = "prior_bounds.npz"
# Default intrinsic prior half-width in Fisher sigmas (config: prior.box_scale).
DEFAULT_BOX_SCALE = 3.0


def cache_path(outdir):
    return os.path.join(outdir, CACHE_NAME)


def _box_from_ingredients(center, sigmas, box_scale):
    """(mins, maxes) for the full 12-D box from the intrinsic linear centre +
    Fisher sigmas at a given scale. The six angle/phase rows are fixed physical
    ranges, independent of the scale."""
    center = np.asarray(center, float)
    sigmas = np.asarray(sigmas, float)
    rows = []
    for k, param in enumerate(INTRINSIC_ORDER):
        # Additive half-width scale*sigma, so a truth of 0 (spin, angles) is fine
        # (a relative width scale*sigma/truth would blow up there).
        half = box_scale * sigmas[k]
        lo, hi = center[k] - half, center[k] + half
        if param in LOG_PARAMS:
            # Keep the log defined if a wide box pokes below zero mass.
            lo = max(lo, 1e-6 * center[k])
            lo, hi = np.log(lo), np.log(hi)
        rows.append([lo, hi])
    rows += [list(r) for r in ANGLE_RANGES]
    mins, maxes = np.array(rows).T
    return mins, maxes


def _build_reparam(reparam_mode, ndim, reparam_idx, cov_intrinsic, truth_vec):
    """Intrinsic-block whitening from the Fisher covariance: physical-space
    (auto), FEW grid-coordinate (grid), or identity (off)."""
    reparam_idx = np.asarray(reparam_idx)
    if reparam_mode in ("auto", "grid"):
        block = cov_intrinsic[np.ix_(reparam_idx, reparam_idx)]
        cls = GridReparam if reparam_mode == "grid" else Reparam
        return cls.from_covariance(ndim, reparam_idx, block,
                                   np.asarray(truth_vec)[reparam_idx])
    return Reparam.identity(ndim, reparam_idx)


def box_ingredients(prec_dict, injection_parameters):
    """(center, sigmas) in INTRINSIC_ORDER, linear coords -- the raw Fisher
    products the box is derived from (cached so the box can be re-scaled on
    load without re-running the Fisher)."""
    center = np.array([injection_parameters[p] for p in INTRINSIC_ORDER], float)
    sigmas = np.array([prec_dict[p] for p in INTRINSIC_ORDER], float)
    return center, sigmas


def build_prior_bounds(prec_dict, cov_lin, cov_order, injection_parameters,
                       truth_vec, reparam_mode, reparam_idx, angle_sigma, ndim,
                       box_scale=DEFAULT_BOX_SCALE):
    """First-run build. Returns (mins, maxes, sample_cov, reparam)."""
    center, sigmas = box_ingredients(prec_dict, injection_parameters)
    mins, maxes = _box_from_ingredients(center, sigmas, box_scale)

    # Correlated proposal from the full Fisher covariance for the intrinsics (a
    # diagonal one drifts along the mass-distance degeneracy into a biased edge
    # mode). cov_lin is in linear coords; map the two mass rows/cols with the
    # 1/mass Jacobian (var[ln m] = var[m]/m^2) to match log-mass sampling.
    assert list(cov_order) == INTRINSIC_ORDER, cov_order
    jac = np.ones(6)
    jac[0] = 1.0 / injection_parameters["mass_1"]
    jac[1] = 1.0 / injection_parameters["mass_2"]
    cov_intrinsic = np.asarray(cov_lin) * np.outer(jac, jac)

    sample_cov = np.zeros((ndim, ndim))
    sample_cov[:6, :6] = cov_intrinsic
    sample_cov[6:, 6:] = np.diag(np.full(ndim - 6, angle_sigma ** 2))

    reparam = _build_reparam(reparam_mode, ndim, reparam_idx, cov_intrinsic, truth_vec)
    return mins, maxes, sample_cov, reparam


def save_prior_bounds(path, mins, maxes, sample_cov, reparam,
                      box_scale=None, prec_dict=None, injection_parameters=None,
                      reparam_mode=None):
    """Cache the box + proposal + reparam. When the Fisher ingredients are given,
    also store box_scale / center / sigmas so a later run with a different
    prior.box_scale can re-derive the box without re-running the Fisher.
    reparam_mode is stored so a resume can detect that its cached transform was
    built in a different coordinate system (auto vs grid)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    extra = {}
    if box_scale is not None and prec_dict is not None and injection_parameters is not None:
        center, sigmas = box_ingredients(prec_dict, injection_parameters)
        extra = dict(box_scale=float(box_scale), box_center=center, box_sigmas=sigmas)
    if reparam_mode is not None:
        extra["reparam_mode"] = str(reparam_mode)
    np.savez(path, mins=mins, maxes=maxes, sample_cov=sample_cov,
             reparam_idx=reparam.idx, reparam_R=reparam.R,
             reparam_sig=reparam.sig, reparam_mu=reparam.mu, **extra)


def load_prior_bounds(path, ndim, reparam_idx, reparam_mode,
                      box_scale=DEFAULT_BOX_SCALE):
    """Resume-path load. Returns (mins, maxes, sample_cov, reparam, reparam_mode).

    If the requested box_scale differs from the cached one and the cache carries
    the Fisher ingredients, the box is re-derived at the new scale (no Fisher
    re-run). Tolerates older caches: a missing sample_cov -> a diagonal
    box-scaled proposal; a missing reparam transform -> reparam_mode downgraded
    to 'off' (so a fresh run is needed to regenerate the whitening).
    """
    cache = np.load(path)
    mins, maxes = cache["mins"], cache["maxes"]

    if "box_scale" in cache.files:
        stored_scale = float(cache["box_scale"])
        if not np.isclose(stored_scale, box_scale):
            mins, maxes = _box_from_ingredients(
                cache["box_center"], cache["box_sigmas"], box_scale)
            logger.info(
                "resuming: cached box_scale=%g != requested %g -> box "
                "re-derived from cached Fisher sigmas", stored_scale, box_scale)
    elif not np.isclose(box_scale, DEFAULT_BOX_SCALE):
        logger.warning(
            "cache predates box_scale support; using its stored box (built at "
            "%g sigma), NOT the requested box_scale=%g. Delete the cache to "
            "rebuild.", DEFAULT_BOX_SCALE, box_scale)
    if "sample_cov" in cache.files:
        sample_cov = cache["sample_cov"]
    else:
        sample_cov = np.diag((0.01 * (maxes - mins)) ** 2)

    if reparam_mode == "off":
        reparam = Reparam.identity(ndim, reparam_idx)
    elif "reparam_R" in cache.files:
        # The cached (R, sig, mu) live in the coordinate system of the mode the
        # cache was built with -- reconstructing them under a different non-off
        # mode would silently mix coordinate systems.
        stored_mode = str(cache["reparam_mode"]) if "reparam_mode" in cache.files else "auto"
        if stored_mode != reparam_mode:
            raise ValueError(
                f"prior-bounds cache at {path} was built with reparam mode "
                f"'{stored_mode}' but this run requests '{reparam_mode}'. "
                f"Delete the cache (or the outdir) to rebuild.")
        cls = GridReparam if reparam_mode == "grid" else Reparam
        reparam = cls(ndim, cache["reparam_idx"], cache["reparam_R"],
                      cache["reparam_sig"], cache["reparam_mu"])
    else:
        logger.warning(
            "resuming: cache has no reparam transform -> falling back to "
            "reparam mode 'off'")
        reparam_mode = "off"
        reparam = Reparam.identity(ndim, reparam_idx)
    return mins, maxes, sample_cov, reparam, reparam_mode
