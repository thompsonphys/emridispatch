"""Toggleable, auto-derived whitening reparametrization for the
intrinsic block.

Modes: "auto" whitens (ln m1, ln m2, a, p, e) via Fisher covariance
in physical coords; "grid" does the same in FEW's kerrecceq grid
coords (more stable across the prior box); "off" is the identity.
"""

from __future__ import annotations

import numpy as np


class Reparam:
    """Affine whitening of a sub-block: ``u = R @ ((x - mu) / sig)``.

    idx selects the transformed indices (rest pass through); R is
    (k,k) orthonormal with rows = axes, wide->tight; sig, mu are the
    block's per-parameter scale and centre.
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
        """Whiten a block from its covariance (in the SAME coords as
        the vector)."""
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

        Stored mode "grid" returns a GridReparam, else the base class.
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

    idx must be the physical block (ln m1, ln m2, a, p, e, dist)
    rows [0..5], mapped through phi before whitening.

    The kerrecceq u coordinate restarts at 0 when p crosses
    p_sep + DELTAPMAX into the far grid region, so phi stitches the
    far branch on above u = 1 as ``1 + k * (U - U0)``, matching value
    and slope at the seam. The stitched u increases monotonically with
    p over the whole grid (0 at the separatrix to ~1.57 at p = 200),
    which also lets phi_inv recover the region from u alone.
    log_abs_det_jac is nonzero; add via ReparamCallable(jacobian=True).
    """

    # --- nonlinear map ------------------------------------------------------
    @staticmethod
    def _maps():
        # Lazy import: keeps emri.reparam importable (diagnostics, notebooks)
        # on machines without few installed.
        from few.utils.geodesic import get_separatrix
        from few.utils.mappings import kerrecceq
        return kerrecceq, get_separatrix

    @staticmethod
    def _separatrix(sep, a, e):
        a = np.asarray(a, float)
        asign = np.sign(a)
        asign[asign == 0.0] = 1.0
        return sep(np.abs(a), np.asarray(e, float), asign)

    @staticmethod
    def _seam(K, pLSO):
        """(U0, k) for the far-region stitch u = 1 + k * (U - U0)."""
        D = K.DELTAPMIN_REGIONB ** -0.5 - (K.PMAX_REGIONB - pLSO) ** -0.5
        U0 = (K.DELTAPMIN_REGIONB ** -0.5 - K.DELTAPMAX ** -0.5) / D
        du_dp = K.ALPHA_FLUX / (2.0 * (K.DELTAPMAX - K.DELTAPMIN) * np.log(2.0))
        return U0, du_dp * D / (0.5 * K.DELTAPMAX ** -1.5)

    @staticmethod
    def phi(block):
        """(ln m1, ln m2, a, p, e, dist) -> (ln m1, ln eta, z, u, w, dist).
        Batch-safe on the last axis."""
        K, sep = GridReparam._maps()
        b = np.atleast_2d(np.asarray(block, float))
        lm1, lm2 = b[:, 0], b[:, 1]
        q = np.exp(lm2 - lm1)                      # mass ratio m2/m1
        ln_eta = np.log(q) - 2.0 * np.log1p(q)     # eta = q/(1+q)^2, stably
        a_, p_, e_ = b[:, 2], b[:, 3], b[:, 4]
        with np.errstate(all="ignore"):
            u, w, y, z, near = K.kerrecceq_forward_map(
                a_, p_, e_, np.ones(len(b)), return_mask=True)
            far = ~np.asarray(near)
            if np.any(far):
                U0, k = GridReparam._seam(
                    K, GridReparam._separatrix(sep, a_[far], e_[far]))
                u = np.asarray(u, float).copy()
                u[far] = 1.0 + k * (u[far] - U0)
        v = np.column_stack([lm1, ln_eta, z, u, w, b[:, 5]])
        return v[0] if np.asarray(block).ndim == 1 else v

    @staticmethod
    def phi_inv(vblock):
        """(ln m1, ln eta, z, u, w, dist) -> (ln m1, ln m2, a, p, e, dist).

        u < 1 inverts the near grid region, u >= 1 the stitched far one.
        Rows with (z, w) outside [0, 1], or u outside the stitched grid
        domain, return NaN instead of raising; log_abs_det_jac then
        returns -inf for them.
        """
        K, sep = GridReparam._maps()
        bwd = K.kerrecceq_backward_map
        v = np.atleast_2d(np.asarray(vblock, float))
        eta = np.exp(v[:, 1])
        # Stable small root of eta*q^2 + (2*eta - 1)*q + eta = 0 (rationalized;
        # the naive quadratic formula loses ~7 digits at EMRI eta ~ 1e-5).
        with np.errstate(all="ignore"):
            q = 2.0 * eta / ((1.0 - 2.0 * eta) + np.sqrt(1.0 - 4.0 * eta))
            lm2 = v[:, 0] + np.log(q)
        z, u, w = v[:, 2], v[:, 3], v[:, 4]
        a = np.full(len(v), np.nan); p = a.copy(); e = a.copy()
        ok = np.isfinite(z) & np.isfinite(u) & np.isfinite(w) \
            & (z >= 0.0) & (z <= 1.0) & (w >= 0.0) & (w <= 1.0)

        near = ok & (u >= 0.0) & (u < 1.0)
        if np.any(near):
            with np.errstate(all="ignore"):
                a[near], p[near], e[near], _x = bwd(
                    u[near], w[near], np.ones(int(near.sum())), z[near],
                    regionA=True)

        far = ok & (u >= 1.0)
        if np.any(far):
            with np.errstate(all="ignore"):
                a_f, _p, e_f, _x = bwd(
                    np.zeros(int(far.sum())), w[far], np.ones(int(far.sum())),
                    z[far], regionA=False)
                U0, k = GridReparam._seam(
                    K, GridReparam._separatrix(sep, a_f, e_f))
                U = U0 + (u[far] - 1.0) / k
                keep = np.flatnonzero(far)[U <= 1.0]
                if keep.size:
                    a[keep], p[keep], e[keep], _x = bwd(
                        U[U <= 1.0], w[keep], np.ones(keep.size), z[keep],
                        regionA=False)
        x = np.column_stack([v[:, 0], lm2, a, p, e, v[:, 5]])
        return x[0] if np.asarray(vblock).ndim == 1 else x

    def __init__(self, ndim, idx, R, sig, mu):
        """(R, sig) whiten v-space; mu = phi(truth block).

        _M = whitening o Jacobian(phi) at the centre, so
        transform_cov maps a physical covariance correctly.
        """
        super().__init__(ndim, idx, R, sig, mu)
        assert list(self.idx) == [0, 1, 2, 3, 4, 5], \
            "GridReparam assumes the standard intrinsic block idx=[0..5]"
        mu_phys = self.phi_inv(self.mu)
        if not np.all(np.isfinite(mu_phys)):
            raise ValueError(
                "reparam mode 'grid': stored centre is outside the kerrecceq "
                "grid domain. Delete the cache (or the outdir) to rebuild.")
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

    @classmethod
    def _check_domain(cls, x_block):
        """Raise unless phi is invertible at a physical block point."""
        K, sep = cls._maps()
        v = cls.phi(x_block)
        if np.all(np.isfinite(v)) and np.allclose(
                cls.phi_inv(v), x_block, rtol=1e-6, atol=1e-9):
            return
        a, p, e = float(x_block[2]), float(x_block[3]), float(x_block[4])
        pLSO = float(np.atleast_1d(cls._separatrix(sep, [a], [e]))[0])
        raise ValueError(
            f"reparam mode 'grid' cannot represent p={p:g}: outside the "
            f"kerrecceq grid domain [{pLSO + K.DELTAPMIN:g}, "
            f"{K.PMAX_REGIONB:g}] at a={a:g}, e={e:g}. Use reparam.mode "
            f"'auto'.")

    # --- factory ------------------------------------------------------------
    @classmethod
    def from_covariance(cls, ndim, idx, cov_block, mu):
        """Whitening built in grid coordinates from a physical-space
        Fisher covariance."""
        idx = np.asarray(idx, dtype=int)
        mu_phys = np.asarray(mu, dtype=float)
        cov_block = np.asarray(cov_block, dtype=float)
        cls._check_domain(mu_phys)
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
        """ln|det dx/du| at u: the varying (nonlinear) part of the
        volume factor.

        = ln|d ln m2/d ln eta| + ln|det d(a,p,e)/d(z,u,w)|; the
        constant whitening determinant is dropped. Returns -inf when
        u maps out of domain.
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
