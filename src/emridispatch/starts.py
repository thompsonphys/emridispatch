"""Cold-chain start points for the EMRI PT-MCMC sampler.

Returns a start in physical coordinates; callers map it through the
reparam transform (to_u) themselves when whitening is on.
"""

import numpy as np


def initial_point(mode, truth_phys, sample_cov_phys, mins, maxes, seed, jitter=5.0):
    """One chain's cold start, in physical coordinates.

    mode: "truth" (the injection), "prior" (uniform draw over
    [mins, maxes]), or "fisher" (truth + jitter sigma-scaled Gaussian,
    clipped to the box). Raises ValueError for any other mode.
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
