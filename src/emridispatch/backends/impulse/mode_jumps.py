"""Adaptive, mode-agnostic jump proposals for impulse PT-MCMC.

The EMRI posterior (in the whitened u-coordinates the sampler runs in) splits
into several modes separated by likelihood barriers. Hot PT chains cross the
barriers; the cold chain's local proposals cannot, so it gets stuck in one mode
and its per-mode weights are wrong. These proposals let ANY chain hop between
modes without hard-coding where the modes are or assuming they share a shape.

The enabling trick: impulse registers a single proposal *instance* on every
temperature chain (ProposalBundle.add_jump adds the same callable to each
chain's JumpProposals), and calls it as ``proposal(chain_stats) -> (q, qxy)``
where ``chain_stats.current_sample`` is that chain's position. So a *stateful*
proposal sees positions from all temperatures over time. We keep a shared
``CrossChainPool`` fed on every call: the hot chains (which visit every mode)
populate it, and the cold chain draws on it to jump. This fixes the root cause
(a stuck chain's own buffer only contains one mode).

Two methods, selectable and stackable:

* ``PopulationDEJump`` (Method 1) -- differential evolution over the shared
  pool: ``q = x + gamma * (Xa - Xb)``. The difference vectors ARE the inter-mode
  offsets, in the right directions and scales, for any geometry / number of
  modes. Symmetric, so ``qxy = 0``. No fitting, no per-mode assumptions.

* ``GMMModeJump`` (Method 2) -- periodically fits a Gaussian mixture (K by BIC)
  to the shared pool over a chosen set of continuous dims, then proposes a
  covariance-aware hop ``x' = mu_t + L_t @ L_c^{-1} @ (x - mu_c)`` from the
  current mode c to a target mode t. This explicitly handles modes with
  DIFFERENT distributions (each has its own mean and covariance). The proposal
  is not symmetric, so it returns the exact log proposal ratio (Tjelmeland &
  Hegstad 2001 mode-jumping form), including the affine Jacobian.

Both learn the modes online -- nothing about location, count, width, or axis is
hard-coded.
"""

from __future__ import annotations

import numpy as np


class CrossChainPool:
    """Fixed-capacity ring buffer of recent positions from all chains.

    Fed on every proposal call (any temperature), so it accumulates samples from
    every mode the hot chains visit. Not thread-safe -- the EMRI run uses
    threads=1 (FEW is not thread-safe), matching impulse's own buffers.
    """

    def __init__(self, ndim: int, capacity: int = 20000):
        self.ndim = ndim
        self.capacity = capacity
        self._buf = np.empty((capacity, ndim))
        self._n = 0        # total pushed
        self._filled = 0   # valid rows

    def push(self, x: np.ndarray) -> None:
        self._buf[self._n % self.capacity] = x
        self._n += 1
        self._filled = min(self._filled + 1, self.capacity)

    def __len__(self) -> int:
        return self._filled

    def sample_two(self, rng) -> tuple[np.ndarray, np.ndarray]:
        i = rng.integers(0, self._filled)
        j = rng.integers(0, self._filled)
        while j == i:
            j = rng.integers(0, self._filled)
        return self._buf[i], self._buf[j]

    def view(self) -> np.ndarray:
        return self._buf[: self._filled]


class PopulationDEJump:
    """Method 1: cross-chain differential-evolution jump. Symmetric (qxy = 0)."""

    __name__ = "popde"  # impulse inspects proposal.__name__ (special-cases 'de')

    def __init__(self, pool: CrossChainPool, min_pool: int = 200,
                 fallback_scale: float = 1e-3):
        self.pool = pool
        self.min_pool = min_pool
        self.fallback_scale = fallback_scale

    def __call__(self, chain_stats):
        rng = chain_stats.rng
        x = chain_stats.current_sample
        # The infinite-temperature chain samples the prior; a mode leap from a
        # prior-wide position lands out of bounds, and on the inf rung the
        # sampler's 1/inf * (-inf lnlike) is a NaN (spurious "invalid value"
        # warnings). It also floods the pool with uniform junk that ruins the
        # GMM fit. So skip it entirely: no proposal, no pool contribution.
        if np.isinf(getattr(chain_stats, "temp", 1.0)):
            return x.copy(), 0.0
        self.pool.push(x)
        q = x.copy()
        if len(self.pool) < self.min_pool:
            # Pool not warm yet: tiny symmetric random walk (harmless, qxy=0).
            q += rng.standard_normal(self.pool.ndim) * self.fallback_scale
            return q, 0.0
        a, b = self.pool.sample_two(rng)
        ndim = self.pool.ndim
        # gamma = 1 half the time (full mode leap), a shrunk optimal-DE scale the
        # other half (within-mode moves). Both symmetric in (a,b) -> qxy = 0.
        gamma = 1.0 if rng.random() < 0.5 else rng.random() * 2.38 / np.sqrt(2 * ndim)
        q = q + gamma * (a - b)
        return q, 0.0


class GMMModeJump:
    """Method 2: adaptive Gaussian-mixture, covariance-aware mode jump.

    Models only ``dims`` (a set of continuous, non-periodic coordinates -- the
    whitened intrinsic block by default). Angle/phase dims are left untouched by
    the jump because they are periodic and not Euclidean-Gaussian. Refits every
    ``refit_every`` calls once the pool holds ``min_pool`` samples.
    """

    __name__ = "gmm_mode"

    def __init__(self, pool: CrossChainPool, dims, max_k: int = 6,
                 refit_every: int = 500, min_pool: int = 500,
                 reg_covar: float = 1e-8, fallback_scale: float = 1e-3,
                 seed: int = 0):
        self.pool = pool
        self.dims = np.asarray(dims, dtype=int)
        self.max_k = max_k
        self.refit_every = refit_every
        self.min_pool = min_pool
        self.reg_covar = reg_covar
        self.fallback_scale = fallback_scale
        self._rng_fit = np.random.default_rng(seed)
        self._calls = 0
        self._last_fit = -1
        # fitted-model cache (None until first successful fit)
        self.gmm = None
        self.mu = self.L = self.Linv = self.logdetL = self.w = None

    def _maybe_refit(self) -> None:
        if len(self.pool) < self.min_pool:
            return
        if self._last_fit >= 0 and (self._calls - self._last_fit) < self.refit_every:
            return
        try:
            from sklearn.mixture import GaussianMixture
        except Exception:
            return  # sklearn missing -> GMM jump silently no-ops (falls back)
        X = self.pool.view()[:, self.dims]
        # Deduplicate exact repeats (MCMC rejections) so BIC/weights reflect
        # distinct support, then pick K by the BIC elbow.
        acc = np.concatenate([[True], np.any(np.diff(X, axis=0) != 0, axis=1)])
        Xi = X[acc]
        if len(Xi) < 5 * self.max_k:
            return
        best, best_bic = None, np.inf
        for k in range(1, self.max_k + 1):
            g = GaussianMixture(k, covariance_type="full", n_init=1,
                                reg_covar=self.reg_covar,
                                random_state=int(self._rng_fit.integers(1 << 30))).fit(Xi)
            b = g.bic(Xi)
            if b < best_bic - 1e-6:
                best_bic, best = b, g
        if best is None:
            return
        # Cache per-component Cholesky factors + log-dets for the affine map.
        covs = best.covariances_
        L = np.linalg.cholesky(covs)
        self.gmm = best
        self.mu = best.means_
        self.L = L
        self.Linv = np.linalg.inv(L)
        self.logdetL = np.log(np.diagonal(L, axis1=1, axis2=2)).sum(axis=1)
        self.w = best.weights_
        self._last_fit = self._calls

    def __call__(self, chain_stats):
        rng = chain_stats.rng
        x = chain_stats.current_sample
        # See PopulationDEJump: skip the infinite-temperature chain (a mode leap
        # from a prior-wide position lands out of bounds -> NaN on the inf rung,
        # and its uniform draws pollute the fitted mixture).
        if np.isinf(getattr(chain_stats, "temp", 1.0)):
            return x.copy(), 0.0
        self.pool.push(x)
        self._calls += 1
        self._maybe_refit()

        q = x.copy()
        if self.gmm is None or len(self.w) < 2:
            q += rng.standard_normal(self.pool.ndim) * self.fallback_scale
            return q, 0.0

        xm = x[self.dims]
        eps = 1e-300
        # Responsibilities of x -> sample the source mode c. If x sits so far
        # from every component that all densities underflow, predict_proba is
        # 0/0 = NaN; bail to a no-op rather than feed NaN probabilities to
        # rng.choice (which would raise).
        r = self.gmm.predict_proba(xm[None])[0]
        if not np.all(np.isfinite(r)):
            return q, 0.0
        c = rng.choice(len(r), p=r)
        # Target mode t: sample from the mixture weights RESTRICTED to t != c, so
        # P(t | c) = w_t / (1 - w_c). (Using raw w_t here would bias the ratio.)
        p_tgt = self.w.copy()
        one_minus_wc = p_tgt.sum() - p_tgt[c]  # = 1 - w_c
        if one_minus_wc <= 0:
            return q, 0.0
        p_tgt[c] = 0.0
        p_tgt = p_tgt / one_minus_wc
        t = rng.choice(len(self.w), p=p_tgt)

        # Covariance-aware affine map: whiten in mode c, un-whiten in mode t.
        xpm = self.mu[t] + self.L[t] @ (self.Linv[c] @ (xm - self.mu[c]))
        q[self.dims] = xpm

        # Exact log proposal ratio (Tjelmeland-Hegstad deterministic mode jump):
        #   forward: source c ~ r_c(x), target t ~ w_t / (1 - w_c);
        #   reverse: source t ~ r_t(x'), target c ~ w_c / (1 - w_t);
        #   affine Jacobian |dx'/dx| = |L_t| / |L_c|.
        rprime = self.gmm.predict_proba(xpm[None])[0]
        one_minus_wt = 1.0 - self.w[t]
        log_j = self.logdetL[t] - self.logdetL[c]
        log_p_fwd = np.log(r[c] + eps) + np.log(self.w[t] + eps) - np.log(one_minus_wc + eps)
        log_p_rev = np.log(rprime[t] + eps) + np.log(self.w[c] + eps) - np.log(one_minus_wt + eps)
        qxy = log_p_rev - log_p_fwd + log_j
        return q, float(qxy)


def build_mode_jumps(method: str, ndim: int, dims, weight: float = 25.0,
                     pool_capacity: int = 20000, seed: int = 0):
    """Return a list of (proposal, weight) for the chosen method, sharing one pool.

    method: "none" | "popde" | "gmm" | "popde+gmm".
    ``dims`` are the coordinates the GMM models.
    """
    pool = CrossChainPool(ndim, capacity=pool_capacity)
    jumps = []
    if "popde" in method:
        jumps.append((PopulationDEJump(pool), weight))
    if "gmm" in method:
        jumps.append((GMMModeJump(pool, dims=dims, seed=seed), weight))
    return jumps, pool
