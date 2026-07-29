"""Per-parameter prior distributions and their 12-D joint composition.

Periodic params wrap into [min, max] before the bounds check; NaN counts
as out of bounds. log_prob is normalized per distribution so mixed prior
types combine correctly; override via a config `priors:` mapping.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from emridispatch.parameters import PARAM_NAMES

__all__ = [
    "Prior", "Uniform", "PeriodicUniform", "LogUniform", "Gaussian",
    "Sine", "Cosine", "CallablePrior", "JointPrior",
    "prior_from_spec", "joint_prior_from_box", "joint_prior_from_config",
    "joint_prior_from_specs",
]


class Prior:
    """Base 1-D prior: bounds, optional period, normalized log_prob, sampler."""

    def __init__(self, minimum, maximum, period=None):
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        if not self.maximum > self.minimum:
            raise ValueError(
                f"prior needs maximum > minimum, got [{minimum}, {maximum}]")
        if period is not None and period <= 0:
            raise ValueError(f"prior period must be positive, got {period}")
        self.period = float(period) if period is not None else None

    @property
    def width(self):
        return self.maximum - self.minimum

    def _inside(self, x):
        return (x >= self.minimum) & (x <= self.maximum)

    def log_prob(self, x):
        raise NotImplementedError

    def sample(self, rng, size=None):
        raise NotImplementedError

    def to_spec(self):
        """Inverse of prior_from_spec: a json-able dict that rebuilds this prior.

        Infinite bounds are omitted (json has no inf; prior_from_spec restores
        the untruncated defaults). PeriodicUniform's period is implied by its
        type, so it is not repeated.
        """
        kind = _SPEC_NAMES.get(type(self))
        if kind is None:
            raise TypeError(
                f"{type(self).__name__} has no spec representation; add it to "
                "_SPEC_NAMES or override to_spec()")
        spec = {"type": kind}
        if np.isfinite(self.minimum):
            spec["min"] = self.minimum
        if np.isfinite(self.maximum):
            spec["max"] = self.maximum
        if self.period is not None and kind != "periodic_uniform":
            spec["period"] = self.period
        return spec

    def __repr__(self):
        args = f"{self.minimum:g}, {self.maximum:g}"
        if self.period is not None:
            args += f", period={self.period:g}"
        return f"{type(self).__name__}({args})"


class Uniform(Prior):
    """Uniform on [min, max]; log_prob = -log(width) inside, -inf outside."""

    def log_prob(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(self._inside(x), -np.log(self.width), -np.inf)[()]

    def sample(self, rng, size=None):
        return rng.uniform(self.minimum, self.maximum, size)


class PeriodicUniform(Uniform):
    """Uniform over one full period of a cyclic coordinate (period = width)."""

    def __init__(self, minimum, maximum):
        super().__init__(minimum, maximum, period=maximum - minimum)


class LogUniform(Prior):
    """p(x) proportional to 1/x on [min, max], min > 0."""

    def __init__(self, minimum, maximum, period=None):
        if minimum <= 0:
            raise ValueError(f"LogUniform needs minimum > 0, got {minimum}")
        super().__init__(minimum, maximum, period)
        self._log_norm = np.log(np.log(self.maximum / self.minimum))

    def log_prob(self, x):
        x = np.asarray(x, dtype=float)
        with np.errstate(all="ignore"):
            lp = -np.log(x) - self._log_norm
        return np.where(self._inside(x), lp, -np.inf)[()]

    def sample(self, rng, size=None):
        return np.exp(rng.uniform(np.log(self.minimum), np.log(self.maximum), size))


class Gaussian(Prior):
    """Gaussian(mu, sigma), optionally truncated to [minimum, maximum]."""

    def __init__(self, mu, sigma, minimum=-np.inf, maximum=np.inf, period=None):
        if sigma <= 0:
            raise ValueError(f"Gaussian needs sigma > 0, got {sigma}")
        self.mu = float(mu)
        self.sigma = float(sigma)
        # Bypass the finite-bounds check in Prior for the untruncated case.
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        if not self.maximum > self.minimum:
            raise ValueError(
                f"prior needs maximum > minimum, got [{minimum}, {maximum}]")
        if period is not None and period <= 0:
            raise ValueError(f"prior period must be positive, got {period}")
        self.period = float(period) if period is not None else None
        # Truncation mass: P(min < X < max) under the untruncated Gaussian.
        za = (self.minimum - self.mu) / self.sigma
        zb = (self.maximum - self.mu) / self.sigma
        self._log_trunc = np.log(ndtr(zb) - ndtr(za))
        self._log_norm = np.log(self.sigma * np.sqrt(2.0 * np.pi))

    def log_prob(self, x):
        x = np.asarray(x, dtype=float)
        z = (x - self.mu) / self.sigma
        lp = -0.5 * z * z - self._log_norm - self._log_trunc
        return np.where(self._inside(x), lp, -np.inf)[()]

    def to_spec(self):
        spec = super().to_spec()
        spec["mu"] = self.mu
        spec["sigma"] = self.sigma
        return spec

    def sample(self, rng, size=None):
        if np.isinf(self.minimum) and np.isinf(self.maximum):
            return rng.normal(self.mu, self.sigma, size)
        # Inverse-CDF truncated sampling.
        za = ndtr((self.minimum - self.mu) / self.sigma)
        zb = ndtr((self.maximum - self.mu) / self.sigma)
        u = rng.uniform(za, zb, size)
        from scipy.special import ndtri

        return self.mu + self.sigma * ndtri(u)


class Sine(Prior):
    """p(x) proportional to sin(x) on [min, max] within [0, pi] (polar angles)."""

    def __init__(self, minimum=0.0, maximum=np.pi, period=None):
        super().__init__(minimum, maximum, period)
        self._norm = np.cos(self.minimum) - np.cos(self.maximum)
        if self._norm <= 0:
            raise ValueError(
                f"Sine prior has non-positive mass on [{minimum}, {maximum}]")

    def log_prob(self, x):
        x = np.asarray(x, dtype=float)
        with np.errstate(all="ignore"):
            lp = np.log(np.sin(x)) - np.log(self._norm)
        return np.where(self._inside(x), lp, -np.inf)[()]

    def sample(self, rng, size=None):
        u = rng.uniform(0.0, 1.0, size)
        return np.arccos(np.cos(self.minimum) - u * self._norm)


class Cosine(Prior):
    """p(x) proportional to cos(x) on [min, max] within [-pi/2, pi/2]."""

    def __init__(self, minimum=-np.pi / 2, maximum=np.pi / 2, period=None):
        super().__init__(minimum, maximum, period)
        self._norm = np.sin(self.maximum) - np.sin(self.minimum)
        if self._norm <= 0:
            raise ValueError(
                f"Cosine prior has non-positive mass on [{minimum}, {maximum}]")

    def log_prob(self, x):
        x = np.asarray(x, dtype=float)
        with np.errstate(all="ignore"):
            lp = np.log(np.cos(x)) - np.log(self._norm)
        return np.where(self._inside(x), lp, -np.inf)[()]

    def sample(self, rng, size=None):
        u = rng.uniform(0.0, 1.0, size)
        return np.arcsin(np.sin(self.minimum) + u * self._norm)


class CallablePrior(Prior):
    """User-supplied log-prob with explicit bounds.

    log_prob_fn(x) is masked to -inf outside bounds; need not handle
    out-of-bounds itself. sample_fn(rng, size) is optional, else sample()
    draws uniform over the bounds.
    """

    def __init__(self, log_prob_fn, minimum, maximum, period=None, sample_fn=None):
        super().__init__(minimum, maximum, period)
        self._log_prob_fn = log_prob_fn
        self._sample_fn = sample_fn

    def log_prob(self, x):
        x = np.asarray(x, dtype=float)
        with np.errstate(all="ignore"):
            lp = np.asarray(self._log_prob_fn(x), dtype=float)
        return np.where(self._inside(x), lp, -np.inf)[()]

    def sample(self, rng, size=None):
        if self._sample_fn is not None:
            return self._sample_fn(rng, size)
        return rng.uniform(self.minimum, self.maximum, size)

    def to_spec(self):
        """Bounds-only spec; the callable itself is not serializable."""
        spec = {"type": "callable"}
        if np.isfinite(self.minimum):
            spec["min"] = self.minimum
        if np.isfinite(self.maximum):
            spec["max"] = self.maximum
        if self.period is not None:
            spec["period"] = self.period
        return spec


_SPEC_TYPES = {
    "uniform": Uniform,
    "periodic_uniform": PeriodicUniform,
    "loguniform": LogUniform,
    "log_uniform": LogUniform,
    "gaussian": Gaussian,
    "normal": Gaussian,
    "sine": Sine,
    "cosine": Cosine,
}

# Canonical spec name per class (inverse of _SPEC_TYPES; aliases collapse).
_SPEC_NAMES = {
    Uniform: "uniform",
    PeriodicUniform: "periodic_uniform",
    LogUniform: "loguniform",
    Gaussian: "gaussian",
    Sine: "sine",
    Cosine: "cosine",
}


def prior_from_spec(spec, default_min=None, default_max=None):
    """Build a Prior from a config dict (e.g. {type: sine}); min/max
    fall back to the given defaults when the spec omits them."""
    spec = dict(spec)
    kind = str(spec.pop("type", "uniform")).lower()
    if kind not in _SPEC_TYPES:
        raise ValueError(
            f"unknown prior type {kind!r}; choose from {sorted(_SPEC_TYPES)}")
    cls = _SPEC_TYPES[kind]

    lo = spec.pop("min", spec.pop("minimum", default_min))
    hi = spec.pop("max", spec.pop("maximum", default_max))
    kwargs = {}
    if kind in ("gaussian", "normal"):
        kwargs["mu"] = float(spec.pop("mu"))
        kwargs["sigma"] = float(spec.pop("sigma"))
        if lo is not None:
            kwargs["minimum"] = float(lo)
        if hi is not None:
            kwargs["maximum"] = float(hi)
    elif kind in ("sine", "cosine"):
        if lo is not None:
            kwargs["minimum"] = float(lo)
        if hi is not None:
            kwargs["maximum"] = float(hi)
    else:
        if lo is None or hi is None:
            raise ValueError(
                f"prior type {kind!r} needs min/max (none available from the box)")
        kwargs["minimum"] = float(lo)
        kwargs["maximum"] = float(hi)
    if "period" in spec and kind not in ("periodic_uniform",):
        kwargs["period"] = float(spec.pop("period"))
    if spec:
        raise ValueError(f"unused keys in prior spec: {sorted(spec)}")
    return cls(**kwargs)


class JointPrior:
    """Joint prior over the sampling vector: a list of independent 1-D Priors.

    Callable returns summed log-prob (-inf out of bounds; periodic params
    are wrapped into range first). Exposes mins/maxes/periodic and the
    per-parameter priors for external samplers.
    """

    def __init__(self, priors, names=None, rng=None):
        self.priors = list(priors)
        self.names = list(names) if names is not None else list(PARAM_NAMES[: len(priors)])
        if len(self.names) != len(self.priors):
            raise ValueError(
                f"{len(self.priors)} priors but {len(self.names)} names")
        self.rng = rng if rng is not None else np.random.default_rng()

    # --- structure -----------------------------------------------------------
    @property
    def ndim(self):
        return len(self.priors)

    @property
    def mins(self):
        return np.array([p.minimum for p in self.priors])

    @property
    def maxes(self):
        return np.array([p.maximum for p in self.priors])

    @property
    def periodic(self):
        """dict index -> period, from per-prior period metadata."""
        return {i: p.period for i, p in enumerate(self.priors) if p.period is not None}

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.priors[self.names.index(key)]
        return self.priors[key]

    def replace(self, name, prior):
        """Return a new JointPrior with the named parameter's prior replaced."""
        priors = list(self.priors)
        priors[self.names.index(name)] = prior
        return JointPrior(priors, self.names, rng=self.rng)

    def spec(self):
        """json-able list of per-parameter spec dicts (name + to_spec()).

        Round-trips through joint_prior_from_specs, except CallablePrior entries
        (bounds-only, flagged type "callable").
        """
        return [{"name": n, **p.to_spec()} for n, p in zip(self.names, self.priors)]

    # --- evaluation ----------------------------------------------------------
    def __call__(self, params):
        params = np.atleast_2d(np.array(params, dtype=float))
        mins, maxes = self.mins, self.maxes

        # Wrap only the parameters declared periodic back into their [min, max]
        # range (modulo the period) before the bounds check.
        for idx, period in self.periodic.items():
            lo = mins[idx]
            col = params[:, idx]
            out_of_wrap = (col < lo) | (col > maxes[idx])
            params[out_of_wrap, idx] = np.mod(col[out_of_wrap] - lo, period) + lo

        out_of_bounds = (
            np.any(params < mins, axis=1)
            | np.any(params > maxes, axis=1)
            | ~np.all(np.isfinite(params), axis=1)
        )

        logp = np.zeros(len(params))
        with np.errstate(all="ignore"):
            for j, prior in enumerate(self.priors):
                logp = logp + np.atleast_1d(prior.log_prob(params[:, j]))
        result = np.where(out_of_bounds | ~np.isfinite(logp), -np.inf, logp)
        return result.squeeze()[()]

    # --- sampling ------------------------------------------------------------
    def sample(self, rng=None, size=None):
        """Draw from the joint prior. Returns (ndim,) or (size, ndim)."""
        rng = rng if rng is not None else self.rng
        draws = [np.atleast_1d(p.sample(rng, size)) for p in self.priors]
        out = np.stack(draws, axis=-1)
        return out[0] if size is None else out

    def initial_sample(self):
        """Alias for sample(); duck-typed callers expect this name."""
        return self.sample()

    def __repr__(self):
        rows = ", ".join(f"{n}={p!r}" for n, p in zip(self.names, self.priors))
        return f"JointPrior({rows})"


def joint_prior_from_box(mins, maxes, periodic_indices=(), names=None,
                         overrides=None):
    """Default JointPrior for a box: Uniform per row, PeriodicUniform for
    periodic_indices, then overrides from a config `priors:` mapping."""
    mins = np.asarray(mins, dtype=float)
    maxes = np.asarray(maxes, dtype=float)
    names = list(names) if names is not None else list(PARAM_NAMES[: len(mins)])
    periodic_indices = set(int(i) for i in periodic_indices)

    priors = [
        PeriodicUniform(lo, hi) if i in periodic_indices else Uniform(lo, hi)
        for i, (lo, hi) in enumerate(zip(mins, maxes))
    ]
    joint = JointPrior(priors, names)

    for name, spec in dict(overrides or {}).items():
        if name not in names:
            raise ValueError(
                f"priors override for unknown parameter {name!r}; "
                f"known: {names}")
        i = names.index(name)
        joint = joint.replace(name, prior_from_spec(
            spec, default_min=mins[i], default_max=maxes[i]))
    return joint


def joint_prior_from_config(cfg, mins, maxes):
    """The JointPrior a run samples under, given its prior box.

    Single source of truth for the box -> prior step: the pipeline builds the
    sampler's prior with it, and the P-P harness draws its truths from it.
    """
    return joint_prior_from_box(mins, maxes, cfg.prior.periodic_2pi_indices,
                                names=PARAM_NAMES, overrides=cfg.priors)


def joint_prior_from_specs(specs):
    """Rebuild a JointPrior from JointPrior.spec() output (e.g. read back from a
    results file). Raises on "callable" entries, whose density was not stored."""
    names, priors = [], []
    for entry in specs:
        entry = dict(entry)
        name = entry.pop("name")
        if str(entry.get("type", "")).lower() == "callable":
            raise ValueError(
                f"prior for {name!r} was a CallablePrior; only its bounds were "
                "stored, the density cannot be reconstructed")
        names.append(name)
        priors.append(prior_from_spec(entry))
    return JointPrior(priors, names)
