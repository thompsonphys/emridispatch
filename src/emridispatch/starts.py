"""Cold-chain start points for the EMRI PT-MCMC sampler.

Pure + testable (no sampler / GPU deps). Returns a start in physical coordinates;
the caller maps it through the reparam transform (to_u) when whitening is on.

Independent cold chains (different seeds) get independent dispersed starts, which
is what makes a cross-chain R-hat meaningful. See the
sampler.start_mode knob in emri_config.yaml.
"""

import numpy as np


def initial_point(mode, truth_phys, sample_cov_phys, mins, maxes, seed, jitter=5.0):
    """One chain's cold start, in physical coordinates.

    Parameters
    ----------
    mode : {"truth", "prior", "fisher"}
        truth  - the injection itself (systematics/recovery mode).
        prior  - uniform draw over the prior box [mins, maxes] (blind-PE dispersion).
        fisher - truth + jitter * Fisher-scale Gaussian, clipped to the box (mild
                 dispersion; the angle block barely crosses modes at this scale).
    truth_phys : array_like
        The injected truth vector (physical coords, masses in log).
    sample_cov_phys : array_like
        The proposal covariance in physical coords (used by "fisher").
    mins, maxes : array_like
        Physical prior-box bounds (used by "prior" and to clip "fisher").
    seed : int
        Seeds the draw so independent chains differ and stay reproducible.
    jitter : float
        Dispersion in Fisher sigmas for "fisher" mode.

    Returns
    -------
    numpy.ndarray
        Length-N physical start vector.
    """
    truth_phys = np.asarray(truth_phys, float)
    mins = np.asarray(mins, float)
    maxes = np.asarray(maxes, float)
    rng = np.random.default_rng(seed)

    if mode == "truth":
        return truth_phys.copy()
    if mode == "prior":
        return rng.uniform(mins, maxes)
    if mode == "fisher":
        ndim = len(truth_phys)
        # Small ridge so a (near-)singular block still factorizes.
        chol = np.linalg.cholesky(np.asarray(sample_cov_phys, float) + 1e-12 * np.eye(ndim))
        x = truth_phys + jitter * (chol @ rng.standard_normal(ndim))
        return np.clip(x, mins, maxes)
    raise ValueError(f"unknown start_mode {mode!r} (truth|prior|fisher)")
