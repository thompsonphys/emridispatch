"""Toggleable, auto-derived whitening reparametrization for the intrinsic block.

The EMRI intrinsic parameters (ln m1, ln m2, a, p, e) are strongly correlated,
so a diagonal proposal mixes across their degeneracy slowly. Rotating that block
into decorrelated coordinates helps, but a whitening fitted once, by hand, to
one injection's chains is fragile: it is anchored to a single point and assumes
the degeneracy direction/scale is the same everywhere (a Gaussian posterior).

This module removes the hard-coding two ways, chosen at runtime:

* ``mode="auto"`` -- build the whitening from the *Fisher covariance* of the
  current injection (``EMRIInjectionGenerator.get_injection_covariance``) plus the
  injection truth as the centre. It is still a single global *linear* map 
  (the local-Gaussian approximation at the truth), 
  so it is slightly less powerful than a map fitted
  to the true, possibly curved/multimodal posterior at that one point.

* ``mode="grid"`` -- like auto, but the Fisher whitening is applied in FEW's
  interpolation-grid coordinates instead of physical ones: the intrinsic block
  (ln m1, ln m2, a, p, e, dist) is first mapped through the non-linear
  (ln m1, ln eta, z, u, w, dist) transform (kerrecceq grid maps + symmetric mass
  ratio), then whitened there. Locally equivalent to auto; globally better: the
  likelihood-ridge orientation rotates ~2x less across a wide prior box in grid
  coordinates (measured on Omega_phi gradients), so one linear whitening stays
  valid over a larger volume, and u in [0,1] respects the separatrix natively.

* ``mode="off"`` -- identity: sample in physical coordinates and let impulse's
  adaptive (AM/SCAM) proposals, seeded with the same Fisher covariance, do the
  decorrelation locally. No anchor, no baked-in transform at all.

The transform is applied only to a sub-block of indices ``idx`` of the full
sampling vector; all other coordinates pass through unchanged. Jacobians: for
"auto" the map is linear, its Jacobian is a constant that cancels in acceptance
ratios, so no term is needed. For "grid" the map is non-linear, and sampling in
u while evaluating the posterior at to_x(u) requires the change-of-variables
term ln|det d x / d u|(u) added to the log-prior -- otherwise the physical-space
posterior is silently multiplied by the (varying) Jacobian. GridReparam provides
log_abs_det_jac(u) and the sampler-side wrapper adds it to the prior only (never
the likelihood, which would double-count).
"""

from __future__ import annotations

import numpy as np


class Reparam:
    """Affine whitening of a sub-block: ``u = R @ ((x - mu) / sig)``.

    Parameters
    ----------
    ndim : int
        Dimension of the full sampling vector.
    idx : array of int
        Indices of the sub-block to transform (the rest pass through).
    R : (k, k) array
        Orthonormal rotation (rows = principal axes, wide -> tight).
    sig : (k,) array
        Per-parameter standardization scales (posterior/Fisher widths).
    mu : (k,) array
        Centre of the block (the frame origin; only shifts coordinates).
    """

    def __init__(self, ndim, idx, R, sig, mu):
        self.ndim = int(ndim)
        self.idx = np.asarray(idx, dtype=int)
        self.R = np.asarray(R, dtype=float)
        self.sig = np.asarray(sig, dtype=float)
        self.mu = np.asarray(mu, dtype=float)
        self._A = self.R / self.sig[None, :]        # x_block -> u_block
        self._Ainv = self.R.T * self.sig[:, None]   # u_block -> x_block
        self._M = np.eye(self.ndim)
        self._M[np.ix_(self.idx, self.idx)] = self._A

    # --- factories -----------------------------------------------------------
    @classmethod
    def identity(cls, ndim, idx=None):
        """A no-op transform (mode='off'): to_u/to_x return their input."""
        idx = np.arange(ndim) if idx is None else np.asarray(idx, dtype=int)
        k = len(idx)
        return cls(ndim, idx, np.eye(k), np.ones(k), np.zeros(k))

    @classmethod
    def from_covariance(cls, ndim, idx, cov_block, mu):
        """Whiten a block from its covariance (in the SAME coords as the vector).

        Standardizes by the marginal sigmas, then rotates into the eigenbasis of
        the resulting correlation matrix (widest/most-degenerate axis first).
        """
        cov_block = np.asarray(cov_block, dtype=float)
        sig = np.sqrt(np.diag(cov_block))
        corr = cov_block / np.outer(sig, sig)
        w, V = np.linalg.eigh(corr)         # ascending eigenvalues
        R = V[:, ::-1].T                     # rows = axes, wide -> tight
        # Deterministic sign: make each row's largest-magnitude entry positive.
        for i in range(R.shape[0]):
            if R[i, np.argmax(np.abs(R[i]))] < 0:
                R[i] = -R[i]
        return cls(ndim, idx, R, sig, np.asarray(mu, dtype=float))

    # --- transforms ----------------------------------------------------------
    def to_u(self, x):
        """Physical -> whitened sampling coords (batch-safe on the last axis)."""
        x = np.array(x, dtype=float)
        u = x.copy()
        u[..., self.idx] = (x[..., self.idx] - self.mu) @ self._A.T
        return u

    def to_x(self, u):
        """Whitened sampling coords -> physical (inverse of to_u)."""
        u = np.array(u, dtype=float)
        x = u.copy()
        x[..., self.idx] = u[..., self.idx] @ self._Ainv.T + self.mu
        return x

    def transform_cov(self, cov):
        """Push a full ndim x ndim covariance through the linear map."""
        return self._M @ np.asarray(cov, dtype=float) @ self._M.T

    def log_abs_det_jac(self, u):
        """ln|det dx/du| at u. Constant for a linear map -> 0 (cancels in MCMC)."""
        return 0.0

    # --- persistence ---------------------------------------------------------
    def save(self, path, mode):
        np.savez(path, mode=mode, idx=self.idx, R=self.R, sig=self.sig, mu=self.mu)

    @classmethod
    def load(cls, path):
        """Return (reparam, mode_str) from a saved transform file.

        Dispatches on the stored mode: "grid" reconstructs a GridReparam (whose
        to_x/to_u include the nonlinear grid map), anything else the base class.
        """
        d = np.load(path, allow_pickle=True)
        mode = str(d["mode"])
        target = GridReparam if mode == "grid" else cls
        # ndim for the full vector isn't stored; callers that need to_u/to_x on
        # full vectors should pass ndim in. For plotting we only touch idx cols,
        # so reconstruct with a generous ndim from the transform matrices.
        rp = target.__new__(target)
        rp.idx = np.asarray(d["idx"], dtype=int)
        rp.R = np.asarray(d["R"], dtype=float)
        rp.sig = np.asarray(d["sig"], dtype=float)
        rp.mu = np.asarray(d["mu"], dtype=float)
        rp._A = rp.R / rp.sig[None, :]
        rp._Ainv = rp.R.T * rp.sig[:, None]
        rp.ndim = int(rp.idx.max() + 1)
        rp._M = None  # not needed for plotting-side inverse
        return rp, mode


class GridReparam(Reparam):
    """Fisher whitening applied in FEW's kerrecceq grid coordinates.

    Physical block x = (ln m1, ln m2, a, p, e, dist) is first mapped through the
    nonlinear ``phi``:

        v = (ln m1, ln eta, z, u_grid, w, dist)

    with (u_grid, w, z) = kerrecceq_forward_map(a, p, e, xI=1) (near/"region A"
    branch -- valid for p < p_sep + 9, which any Fisher-scaled box satisfies) and
    eta the symmetric mass ratio. The affine whitening (R, sig, mu) then acts on
    v, with mu = phi(truth block). ``idx`` must be the standard intrinsic block
    [0..5]: the row semantics above are baked in.

    The map is non-linear, so ``log_abs_det_jac`` is nonzero and must be added to
    the log-prior by the sampling wrapper (ReparamCallable(jacobian=True)).
    Out-of-domain u (backward map poked outside the grid) yields NaN physical
    values; the prior box test rejects those points.
    """

    # --- nonlinear map ------------------------------------------------------
    @staticmethod
    def _maps():
        # Lazy import: keeps emri.reparam importable (diagnostics, notebooks)
        # on machines without few installed.
        from few.utils.mappings.kerrecceq import (
            kerrecceq_forward_map, kerrecceq_backward_map)
        return kerrecceq_forward_map, kerrecceq_backward_map

    @staticmethod
    def phi(block):
        """(ln m1, ln m2, a, p, e, dist) -> (ln m1, ln eta, z, u, w, dist).
        Batch-safe on the last axis."""
        fwd, _ = GridReparam._maps()
        b = np.atleast_2d(np.asarray(block, float))
        lm1, lm2 = b[:, 0], b[:, 1]
        q = np.exp(lm2 - lm1)                      # mass ratio m2/m1
        ln_eta = np.log(q) - 2.0 * np.log1p(q)     # eta = q/(1+q)^2, stably
        with np.errstate(all="ignore"):
            u, w, y, z = fwd(b[:, 2], b[:, 3], b[:, 4], np.ones(len(b)))
        v = np.column_stack([lm1, ln_eta, z, u, w, b[:, 5]])
        return v[0] if np.asarray(block).ndim == 1 else v

    @staticmethod
    def phi_inv(vblock):
        """(ln m1, ln eta, z, u, w, dist) -> (ln m1, ln m2, a, p, e, dist).

        Rows whose (z, u, w) fall outside the grid domain [0, 1] come back NaN
        instead of being passed to the backward map (whose separatrix root-find
        raises on garbage). NaN rows are rejected by
        log_abs_det_jac (-inf on the prior), so wild proposals die cleanly.
        """
        _, bwd = GridReparam._maps()
        v = np.atleast_2d(np.asarray(vblock, float))
        eta = np.exp(v[:, 1])
        # Stable small root of eta*q^2 + (2*eta - 1)*q + eta = 0 (rationalized;
        # the naive quadratic formula loses ~7 digits at EMRI eta ~ 1e-5).
        with np.errstate(all="ignore"):
            q = 2.0 * eta / ((1.0 - 2.0 * eta) + np.sqrt(1.0 - 4.0 * eta))
            lm2 = v[:, 0] + np.log(q)
        zuw = v[:, 2:5]
        ok = np.all(np.isfinite(zuw), axis=1) & np.all(zuw >= 0.0, axis=1) \
            & np.all(zuw <= 1.0, axis=1)
        a = np.full(len(v), np.nan); p = a.copy(); e = a.copy()
        if np.any(ok):
            with np.errstate(all="ignore"):
                a[ok], p[ok], e[ok], _x = bwd(v[ok, 3], v[ok, 4],
                                              np.ones(int(ok.sum())), v[ok, 2],
                                              regionA=True)
        x = np.column_stack([v[:, 0], lm2, a, p, e, v[:, 5]])
        return x[0] if np.asarray(vblock).ndim == 1 else x

    def __init__(self, ndim, idx, R, sig, mu):
        """(R, sig) whiten v-space; mu = phi(truth block). Rebuilds the full-dim
        linear approximation _M = whitening o (numerical Jacobian of phi at the
        centre), so transform_cov maps a physical covariance correctly."""
        super().__init__(ndim, idx, R, sig, mu)
        assert list(self.idx) == [0, 1, 2, 3, 4, 5], \
            "GridReparam assumes the standard intrinsic block idx=[0..5]"
        mu_phys = self.phi_inv(self.mu)
        J = self._phi_jacobian(mu_phys)
        self._M = np.eye(self.ndim)
        self._M[np.ix_(self.idx, self.idx)] = self._A @ J

    @staticmethod
    def _phi_jacobian(x_block, rel_step=1e-6, abs_step=1e-8):
        """Central-difference Jacobian d phi / d x at a physical block point."""
        x_block = np.asarray(x_block, float)
        J = np.zeros((6, 6))
        for j in range(6):
            h = max(abs_step, rel_step * abs(x_block[j]))
            xp_, xm_ = x_block.copy(), x_block.copy()
            xp_[j] += h; xm_[j] -= h
            J[:, j] = (GridReparam.phi(xp_) - GridReparam.phi(xm_)) / (2.0 * h)
        return J

    # --- factory ------------------------------------------------------------
    @classmethod
    def from_covariance(cls, ndim, idx, cov_block, mu):
        """Whitening built in grid coordinates: push the physical-space Fisher
        covariance through the numerical Jacobian of phi at the truth, whiten
        there, centre at phi(truth)."""
        idx = np.asarray(idx, dtype=int)
        mu_phys = np.asarray(mu, dtype=float)
        cov_block = np.asarray(cov_block, dtype=float)
        J = cls._phi_jacobian(mu_phys)
        cov_v = J @ cov_block @ J.T
        base = Reparam.from_covariance(ndim, idx, cov_v, cls.phi(mu_phys))
        return cls(ndim, idx, base.R, base.sig, base.mu)

    # --- transforms ----------------------------------------------------------
    def to_u(self, x):
        x = np.array(x, dtype=float)
        u = x.copy()
        u[..., self.idx] = (self.phi(x[..., self.idx]) - self.mu) @ self._A.T
        return u

    def to_x(self, u):
        u = np.array(u, dtype=float)
        x = u.copy()
        v = u[..., self.idx] @ self._Ainv.T + self.mu
        x[..., self.idx] = self.phi_inv(v)
        return x

    def log_abs_det_jac(self, u):
        """ln|det dx/du| at u: the varying (nonlinear) part of the volume factor.

        Factorizes as |d ln m2 / d ln eta| * |det d(a,p,e)/d(z,u_grid,w)|; the
        constant linear-whitening determinant cancels in MCMC and is dropped.
        Computed by central differences of phi_inv in v-space (grid-map formulae
        are closed-form and cheap). Returns -inf when the point maps out of
        domain (NaNs), so such proposals are rejected cleanly at the prior.
        """
        u = np.asarray(u, dtype=float)
        v = u[self.idx] @ self._Ainv.T + self.mu
        # analytic mass factor: d ln m2 / d ln eta = (1+q)/(1-q)
        eta = np.exp(v[1])
        root = np.sqrt(max(1.0 - 4.0 * eta, 0.0))
        q = 2.0 * eta / ((1.0 - 2.0 * eta) + root) if root > 0 else 1.0
        if not (0.0 < q < 1.0):
            return -np.inf
        mass_term = np.log((1.0 + q) / (1.0 - q))
        # numeric 3x3 block d(a,p,e)/d(z,u_grid,w) at v
        h = 1e-6
        Jg = np.zeros((3, 3))
        for j, col in enumerate((2, 3, 4)):        # v rows: z, u_grid, w
            vp, vm = v.copy(), v.copy()
            vp[col] += h; vm[col] -= h
            xp_ = self.phi_inv(vp); xm_ = self.phi_inv(vm)
            Jg[:, j] = (xp_[2:5] - xm_[2:5]) / (2.0 * h)
        if not np.all(np.isfinite(Jg)):
            return -np.inf
        sgn, logdet = np.linalg.slogdet(Jg)
        if sgn == 0:
            return -np.inf
        return mass_term + logdet


class ReparamCallable:
    """Wrap a lnlike/lnprior so the sampler works in u-space: each proposed u is
    mapped back to physical x before the wrapped callable sees it.

    jacobian=True additionally adds ln|det dx/du|(u) to the wrapped value -- set
    it on the prior wrapper (and only there) when the reparam is nonlinear, so
    the u-space stationary density is the correctly transformed posterior."""

    def __init__(self, inner, reparam: Reparam, jacobian=False):
        self.inner = inner
        self.reparam = reparam
        self.jacobian = jacobian

    def __call__(self, params):
        val = self.inner(self.reparam.to_x(params))
        if self.jacobian:
            val = val + self.reparam.log_abs_det_jac(params)
        return val

    def initial_sample(self):
        return self.reparam.to_u(self.inner.initial_sample())

    def __getattr__(self, name):
        return getattr(self.inner, name)
