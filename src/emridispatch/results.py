"""Sampler-agnostic results: raw backend output -> common HDF5 format.

Converts a backend outdir (impulse chain_N.txt; eryn eryn_chain.h5)
into one format_version 1 results.h5. Requires h5py
(pip install emridispatch[results]).

lnprob is the tempered log posterior, lnprior + beta*lnlike, the density each
rung actually samples. That is impulse's native convention; eryn stores an
untempered log prior, so the converter rebuilds the tempered value from the
per-step betas. lnprior is stored alongside it, so the untempered log posterior
is lnprior + lnlike for either backend. Rung 0 has beta = 1, making the two
coincide there.

Both backends can adapt their ladder mid-run, so betas is a full
(nsteps, ntemps) history rather than one vector; temperatures holds only the
final ladder. When the ladder moved, a hot rung's samples are a ladder slot of
drifting temperature rather than a fixed-temperature chain, and thermodynamic
evidence estimates are invalid. Results.ladder_adapted() reports this. The cold
rung is always safe because beta = 1 is pinned.

Chain axis 1 holds ndraws = nsteps * nwalkers rows, step-major, pooling an
ensemble backend's walkers into one sample set. nwalkers is stored so the
(nsteps, nwalkers) factorization stays recoverable: it is what lets burn and
thin count time steps rather than rows, and what lets diagnostics rebuild
per-walker time series. Non-ensemble backends store nwalkers = 1, making
ndraws == nsteps.

Schema; root attrs format_version, backend, ndim, ntemps, nwalkers, nsteps,
ndraws, param_names (json), config (json):
    /chains/samples    (ntemps, ndraws, ndim)  raw sampling coords
    /chains/lnlike     (ntemps, ndraws)        untempered
    /chains/lnprior    (ntemps, ndraws)
    /chains/lnprob     (ntemps, ndraws)        tempered: lnprior + beta*lnlike
    /chains/accepted   (ntemps, ndraws)        backend-specific, see converters
    /betas             (nsteps, ntemps)        ladder history, 0 allowed
    /temperatures      (ntemps,)               final ladder; inf allowed
    /physical/samples  (ntemps, ndraws, ndim)  whitening inverted
    /truth             sampling_vector, physical_vector; injection json attr
    /meta              config_yaml, run_log, run_summary verbatim
    /prior             spec json attr plus prior_bounds.npz arrays
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
from dataclasses import dataclass, field

import numpy as np

from emridispatch.parameters import NDIM, PARAM_NAMES

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
DEFAULT_NAME = "results.h5"

# Relative beta drift below which an adapting ladder is treated as fixed.
LADDER_DRIFT_RTOL = 1.0e-2

# Raw impulse chain columns after the ndim parameters.
_EXTRA_COLS = 4  # lnlike, lnprob, accepted, temperature


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "results files need h5py; install with `pip install emridispatch[results]`"
        ) from exc
    return h5py


@dataclass
class Results:
    """In-memory view of one run's chains + metadata (see module schema)."""

    samples: np.ndarray            # (ntemps, ndraws, ndim) sampling coords
    lnlike: np.ndarray             # (ntemps, ndraws)
    lnprior: np.ndarray            # (ntemps, ndraws)
    lnprob: np.ndarray             # (ntemps, ndraws)
    accepted: np.ndarray           # (ntemps, ndraws)
    betas: np.ndarray              # (nsteps, ntemps)
    temperatures: np.ndarray       # (ntemps,)
    nwalkers: int = 1
    param_names: list = field(default_factory=lambda: list(PARAM_NAMES))
    physical: np.ndarray | None = None   # (ntemps, ndraws, ndim) or None
    truth_sampling: np.ndarray | None = None
    truth_physical: np.ndarray | None = None
    injection: dict | None = None
    prior_spec: list | None = None       # JointPrior.spec() output
    prior_arrays: dict = field(default_factory=dict)   # prior_bounds.npz arrays
    prior_meta: dict = field(default_factory=dict)     # its scalars/strings
    backend: str = ""
    config: dict = field(default_factory=dict)
    config_yaml: str | None = None       # verbatim config.yaml text
    run_log: str | None = None           # verbatim run.log text
    run_summary: str | None = None       # verbatim run_summary.json text

    def __post_init__(self):
        """Pin the betas layout, which is transposed relative to the chains.

        betas_per_draw() would otherwise silently under-return on a mismatch
        instead of failing.
        """
        want = (self.nsteps, self.ntemps)
        if np.shape(self.betas) != want:
            raise ValueError(
                f"betas has shape {np.shape(self.betas)}, expected {want} "
                "(nsteps, ntemps)")

    # --- structure -----------------------------------------------------------
    @property
    def ntemps(self):
        return self.samples.shape[0]

    @property
    def ndraws(self):
        """Stored rows per rung: nsteps * nwalkers."""
        return self.samples.shape[1]

    @property
    def nsteps(self):
        """Markov time steps per rung, with an ensemble's walkers folded out."""
        return self.samples.shape[1] // self.nwalkers

    @property
    def ndim(self):
        return self.samples.shape[2]

    def ladder_drift(self):
        """(ntemps,) largest relative spread of each rung's beta over the run."""
        lo, hi = self.betas.min(axis=0), self.betas.max(axis=0)
        scale = np.where(hi == 0.0, 1.0, np.abs(hi))
        return (hi - lo) / scale

    def ladder_adapted(self, rtol=LADDER_DRIFT_RTOL):
        """True when the temperature ladder moved enough to matter.

        A hot rung is then a ladder slot of drifting temperature rather than a
        fixed-temperature chain, so `temperatures` labels it only by where the
        ladder ended up, and thermodynamic evidence estimates are invalid. The
        cold rung is unaffected: beta = 1 is pinned.

        The tolerance is deliberately loose. eryn adapts by default and never
        settles exactly, so a sub-percent wobble is not worth relabelling a rung
        over -- a strict comparison flags essentially every tempered run.
        """
        return bool(np.any(self.ladder_drift() > rtol))

    def betas_per_draw(self):
        """(ntemps, ndraws) inverse temperatures aligned with the draw axis.

        Broadcasts each step's ladder across that step's walkers, matching the
        step-major layout of the stored chains.
        """
        b = np.repeat(self.betas.T[:, :, None], self.nwalkers, axis=2)
        return b.reshape(self.ntemps, -1)[:, :self.ndraws]

    def rung_temperatures(self, i):
        """(nsteps,) temperature history of one rung; inf where beta is 0."""
        return _temperatures_from_betas(self.betas[:, i])

    def _steps(self, data, i):
        """One rung's rows regrouped as (nsteps, nwalkers, ndim)."""
        s = data[i]
        keep = len(s) - len(s) % self.nwalkers
        return s[:keep].reshape(-1, self.nwalkers, s.shape[-1])

    # --- accessors -----------------------------------------------------------
    def rung(self, i, physical=True, burn=0, thin=1):
        """(nkept, ndim) samples of one rung, walkers pooled.

        burn and thin count time steps, not stored rows: one step of an
        ensemble backend contributes nwalkers rows, and both are applied on the
        step axis, so thinning drops whole steps instead of selecting a walker
        subset. This matches the burn units used by
        emridispatch.diagnostics.stack_chains. thin is cosmetic (plot
        rendering), not a statistical device -- quote MC error via ESS instead.
        """
        data = self.physical if physical else self.samples
        if data is None:
            raise ValueError(
                "no physical-coordinate samples stored (conversion had no "
                "reparam transform); use physical=False")
        kept = self._steps(data, i)[burn:][::thin]
        if len(kept) < 2:
            raise ValueError(
                f"after burn={burn} and thin={thin} only {len(kept)} of this "
                f"run's {self.nsteps} time step(s) remain; reduce --burn or "
                "--thin")
        return kept.reshape(-1, data.shape[-1])

    def posterior(self, physical=True, burn=0, thin=1):
        """Cold-chain (rung 0) samples -- the posterior."""
        return self.rung(0, physical=physical, burn=burn, thin=thin)

    def prior(self):
        """Reconstruct the run's JointPrior from the stored spec (prior draws,
        log-prob evaluation, reweighting)."""
        if self.prior_spec is None:
            raise ValueError(
                "no prior spec stored (run predates prior_spec.json); only the "
                f"box arrays are available: {sorted(self.prior_arrays)}")
        from emridispatch.priors import joint_prior_from_specs

        return joint_prior_from_specs(self.prior_spec)

    # --- persistence ---------------------------------------------------------
    def save(self, path):
        h5py = _require_h5py()
        with h5py.File(path, "w") as f:
            f.attrs["format_version"] = FORMAT_VERSION
            f.attrs["backend"] = self.backend
            f.attrs["ndim"] = self.ndim
            f.attrs["ntemps"] = self.ntemps
            f.attrs["nwalkers"] = self.nwalkers
            f.attrs["nsteps"] = self.nsteps
            f.attrs["ndraws"] = self.ndraws
            f.attrs["param_names"] = json.dumps(list(self.param_names))
            f.attrs["config"] = json.dumps(self.config)

            g = f.create_group("chains")
            g.create_dataset("samples", data=self.samples)
            g.create_dataset("lnlike", data=self.lnlike)
            g.create_dataset("lnprior", data=self.lnprior)
            g.create_dataset("lnprob", data=self.lnprob)
            g.create_dataset("accepted", data=self.accepted)
            f.create_dataset("betas", data=self.betas)
            f.create_dataset("temperatures", data=self.temperatures)

            if self.physical is not None:
                f.create_group("physical").create_dataset(
                    "samples", data=self.physical)

            if (self.truth_sampling is not None
                    or self.truth_physical is not None
                    or self.injection is not None):
                t = f.create_group("truth")
                if self.truth_sampling is not None:
                    t.create_dataset("sampling_vector", data=self.truth_sampling)
                if self.truth_physical is not None:
                    t.create_dataset("physical_vector", data=self.truth_physical)
                if self.injection is not None:
                    t.attrs["injection"] = json.dumps(self.injection)

            meta = {"config_yaml": self.config_yaml, "run_log": self.run_log,
                    "run_summary": self.run_summary}
            if any(v is not None for v in meta.values()):
                m = f.create_group("meta")
                for k, v in meta.items():
                    if v is not None:
                        m.create_dataset(k, data=v)

            if self.prior_spec is not None or self.prior_arrays:
                p = f.create_group("prior")
                if self.prior_spec is not None:
                    p.attrs["spec"] = json.dumps(self.prior_spec)
                for k, v in self.prior_arrays.items():
                    p.create_dataset(k, data=v)
                for k, v in self.prior_meta.items():
                    p.attrs[k] = v
        return path

    @classmethod
    def load(cls, path):
        h5py = _require_h5py()
        with h5py.File(path, "r") as f:
            version = int(f.attrs.get("format_version", -1))
            if version != FORMAT_VERSION:
                raise ValueError(
                    f"{path}: results format_version {version} not supported "
                    f"(expected {FORMAT_VERSION}); regenerate it with "
                    f"`emridisp-postprocess {os.path.dirname(path) or '.'} -f`")
            kw = dict(
                samples=f["chains/samples"][()],
                lnlike=f["chains/lnlike"][()],
                lnprior=f["chains/lnprior"][()],
                lnprob=f["chains/lnprob"][()],
                accepted=f["chains/accepted"][()],
                betas=f["betas"][()],
                temperatures=f["temperatures"][()],
                nwalkers=int(f.attrs["nwalkers"]),
                param_names=json.loads(f.attrs["param_names"]),
                backend=str(f.attrs["backend"]),
                config=json.loads(f.attrs["config"]),
            )
            if "physical" in f:
                kw["physical"] = f["physical/samples"][()]
            if "truth" in f:
                t = f["truth"]
                if "sampling_vector" in t:
                    kw["truth_sampling"] = t["sampling_vector"][()]
                if "physical_vector" in t:
                    kw["truth_physical"] = t["physical_vector"][()]
                if "injection" in t.attrs:
                    kw["injection"] = json.loads(t.attrs["injection"])
            if "meta" in f:
                m = f["meta"]
                for k in ("config_yaml", "run_log", "run_summary"):
                    if k in m:
                        kw[k] = m[k].asstr()[()]
            if "prior" in f:
                p = f["prior"]
                if "spec" in p.attrs:
                    kw["prior_spec"] = json.loads(p.attrs["spec"])
                kw["prior_arrays"] = {k: p[k][()] for k in p.keys()}
                kw["prior_meta"] = {k: p.attrs[k] for k in p.attrs
                                    if k != "spec"}
        return cls(**kw)


# --------------------------------------------------------------------------
# Converter registry: raw backend outdir -> Results
# --------------------------------------------------------------------------

_CONVERTERS = {}


def register_converter(name):
    """Decorator: register fn(run_dir) -> Results under a backend name."""
    def deco(fn):
        _CONVERTERS[name] = fn
        return fn
    return deco


def detect_backend(run_dir):
    """Backend name from run_summary.json, falling back to the config.yaml the
    CLI copies into the outdir (killed runs have no summary). None if neither
    is available."""
    path = os.path.join(run_dir, "run_summary.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh).get("config", {}).get("backend")
    yaml_path = os.path.join(run_dir, "config.yaml")
    if os.path.exists(yaml_path):
        import yaml

        with open(yaml_path) as fh:
            raw = yaml.safe_load(fh) or {}
        # Same default as pipeline.run_from_config when sampler.backend unset.
        return (raw.get("sampler") or {}).get("backend", "impulse")
    return None


def is_complete(run_dir):
    """True iff the run dir holds a finished run."""
    return os.path.exists(os.path.join(run_dir, "run_summary.json"))


def load_or_convert(run_dir, backend=None):
    """Results for a run dir: the saved results.h5 if present, else an
    in-memory convert() of the raw files."""
    path = os.path.join(run_dir, DEFAULT_NAME)
    if os.path.exists(path):
        return Results.load(path)
    return convert(run_dir, backend=backend)


def convert(run_dir, backend=None):
    """Convert one raw run dir into a Results via the registered converter."""
    if backend is None:
        backend = detect_backend(run_dir)
        if backend is None:
            raise ValueError(
                f"{run_dir}: no run_summary.json or config.yaml to auto-detect "
                "the backend from; pass --backend explicitly")
    if backend not in _CONVERTERS:
        raise ValueError(
            f"no converter for backend {backend!r}; known: {sorted(_CONVERTERS)}")
    results = _CONVERTERS[backend](run_dir)
    _attach_run_files(results, run_dir)
    return results


def _attach_run_files(results, run_dir):
    """Embed verbatim config.yaml and run.log from the outdir."""
    cfg_path = os.path.join(run_dir, "config.yaml")
    if results.config_yaml is None and os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            results.config_yaml = fh.read()

    log_name = "run.log"
    if results.config_yaml is not None:
        try:
            import yaml

            raw = yaml.safe_load(results.config_yaml) or {}
            log_name = (raw.get("logging") or {}).get("file", "run.log")
        except Exception:
            pass
    log_path = os.path.join(run_dir, log_name)
    if results.run_log is None and os.path.exists(log_path):
        with open(log_path, errors="replace") as fh:
            results.run_log = fh.read()

    summary_path = os.path.join(run_dir, "run_summary.json")
    if results.run_summary is None and os.path.exists(summary_path):
        with open(summary_path) as fh:
            results.run_summary = fh.read()


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _load_sidecars(run_dir, samples):
    """Backend-agnostic sidecar files -> shared Results kwargs.

    Reads reparam_transform.npz, injection_truth.json, prior_spec.json /
    prior_bounds.npz, and run_summary.json (config.yaml fallback).
    `samples` must be an (..., NDIM) array in raw sampling coordinates.
    """
    # Invert the whitening to physical sampling-vector coordinates.
    reparam = None
    rp_path = os.path.join(run_dir, "reparam_transform.npz")
    if os.path.exists(rp_path):
        from emridispatch.reparam import Reparam

        reparam, _mode = Reparam.load(rp_path)
        flat = samples.reshape(-1, NDIM)
        physical = reparam.to_x(flat).reshape(samples.shape)
    else:
        logger.info("no reparam_transform.npz; sampling coords are physical")
        physical = samples.copy()

    # Truth: injection_truth.json's sampling_vector is in physical
    # sampling-vector coordinates (ln masses); map forward for the raw frame.
    truth_physical = truth_sampling = injection = None
    truth = _load_json(os.path.join(run_dir, "injection_truth.json"))
    if truth is not None:
        truth_physical = np.asarray(truth["sampling_vector"], dtype=float)
        truth_sampling = (reparam.to_u(truth_physical)
                          if reparam is not None else truth_physical.copy())
        injection = truth.get("injection")

    # Prior: exact spec when the run wrote it, box arrays either way.
    prior_spec = _load_json(os.path.join(run_dir, "prior_spec.json"))
    prior_arrays, prior_meta = {}, {}
    pb_path = os.path.join(run_dir, "prior_bounds.npz")
    if os.path.exists(pb_path):
        with np.load(pb_path, allow_pickle=True) as pb:
            for k in pb.files:
                v = pb[k]
                if v.ndim == 0:
                    prior_meta[k] = v.item() if v.dtype.kind in "fiub" else str(v)
                else:
                    prior_arrays[k] = v

    summary = _load_json(os.path.join(run_dir, "run_summary.json"))
    config = (summary or {}).get("config", {})
    if summary is None:
        # Killed/incomplete run: fall back to the config copied into the outdir
        # by the CLI (config.yaml) for at least the run's configuration.
        yaml_path = os.path.join(run_dir, "config.yaml")
        if os.path.exists(yaml_path):
            import yaml

            with open(yaml_path) as fh:
                config = yaml.safe_load(fh) or {}
            logger.warning("%s: no run_summary.json (incomplete run?); using "
                           "config.yaml for config metadata", run_dir)
        else:
            logger.warning("%s: no run_summary.json (incomplete run?); "
                           "converting anyway with empty config metadata",
                           run_dir)

    return dict(
        physical=physical, truth_sampling=truth_sampling,
        truth_physical=truth_physical, injection=injection,
        prior_spec=prior_spec, prior_arrays=prior_arrays,
        prior_meta=prior_meta, config=config,
    )


def _betas_from_temperatures(temperatures):
    """Inverse temperatures; T = inf maps to the legitimate beta = 0.

    Rejects non-positive and NaN temperatures rather than inverting them: a
    T = 0 rung would give beta = inf, which then turns every likelihood term
    into NaN further down instead of failing here.
    """
    t = np.asarray(temperatures, dtype=float)
    if not np.all(t > 0.0):
        raise ValueError(
            "temperature ladder has entries that are not positive: "
            f"{np.unique(t[~(t > 0.0)]).tolist()}; temperatures must be > 0 "
            "(inf is allowed and means beta = 0)")
    return 1.0 / t


def _temperatures_from_betas(betas):
    """Temperatures; the legitimate beta = 0 maps back to T = inf.

    Anything else, including a negative or NaN beta, is inverted as-is rather
    than being folded into inf, so a corrupt ladder stays visible.
    """
    b = np.asarray(betas, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(b == 0.0, np.inf, 1.0 / b)


def _temper(lnprior, betas, lnlike):
    """lnprior + beta*lnlike, with a beta = 0 rung contributing no likelihood.

    An infinite-temperature rung samples the prior, so its tempered posterior
    is lnprior exactly. Evaluating that as 0*lnlike would give NaN wherever the
    likelihood is -inf, discarding a value that is known.
    """
    zero = betas == 0.0
    safe = np.where(zero, 1.0, betas)
    return np.where(zero, lnprior, lnprior + safe * lnlike)


def _untemper(lnprob, betas, lnlike, run_dir):
    """Recover lnprior from a tempered lnprob, avoiding 0*inf and inf-inf.

    A beta = 0 rung samples the prior, so lnprior is lnprob exactly. Where
    beta > 0 and the likelihood is -inf, lnprob is -inf too and carries no
    information about lnprior; that is genuinely unrecoverable, so it is
    counted and reported rather than left as a silent NaN.
    """
    upstream = np.isnan(lnprob)
    if upstream.any():
        logger.warning(
            "%s: %d of %d lnprob value(s) are already NaN in the raw chain, so "
            "the prior is unrecoverable there. The sampler tempers as "
            "lnprior + lnlike/temp, which is NaN for a -inf likelihood on an "
            "infinite-temperature rung", run_dir, int(upstream.sum()),
            lnprob.size)

    zero = betas == 0.0
    safe = np.where(zero, 1.0, betas)
    with np.errstate(invalid="ignore"):
        lnprior = np.where(zero, lnprob, lnprob - safe * lnlike)
    lost = ~np.isfinite(lnprior) & ~zero & ~upstream
    if lost.any():
        logger.warning(
            "%s: %d of %d draw(s) have a non-finite tempered lnprob, which "
            "carries no information about lnprior; storing NaN there. This is "
            "a -inf likelihood on a finite-temperature rung, usually an "
            "out-of-domain waveform", run_dir, int(lost.sum()), lnprior.size)
    return lnprior


def _warn_if_ladder_adapted(results, run_dir):
    """Flag a ladder that moved mid-run, which mislabels the hot rungs."""
    if results.ladder_adapted():
        logger.warning(
            "%s: the temperature ladder adapted during the run; "
            "/temperatures records only the final ladder, so hot-rung samples "
            "span a range of temperatures and are not a fixed-temperature "
            "chain. The cold rung (beta = 1) is unaffected. Freeze the ladder "
            "(eryn stop_adaptation, impulse ladder.adapt: false) if you need "
            "per-rung temperatures or thermodynamic evidence", run_dir)


@register_converter("impulse")
def _convert_impulse(run_dir):
    """impulse PTSampler raw output: chain_N.txt per rung, cols
    [params(ndim), lnlike, lnprob, accepted, temperature].

    impulse's lnprob column is already tempered (lnprior + lnlike/temp), which
    is the schema's convention, so lnprior is recovered by subtracting the
    tempered likelihood term. The per-step temperature column doubles as the
    ladder history, which is not constant when sampler.impulse.ladder.adapt
    is on.
    """
    paths = glob.glob(os.path.join(run_dir, "chain_*.txt"))
    indexed = []
    for p in paths:
        m = re.fullmatch(r"chain_(\d+)\.txt", os.path.basename(p))
        if m:
            indexed.append((int(m.group(1)), p))
    if not indexed:
        raise FileNotFoundError(f"{run_dir}: no chain_N.txt files found")
    indexed.sort()

    arrs = [np.loadtxt(p, ndmin=2) for _, p in indexed]
    ncols = arrs[0].shape[1]
    if ncols < NDIM + _EXTRA_COLS:
        raise ValueError(
            f"{run_dir}: chain files have {ncols} columns, expected at least "
            f"{NDIM + _EXTRA_COLS}")
    nsteps = min(len(a) for a in arrs)
    if nsteps < 1:
        raise ValueError(f"{run_dir}: empty chain files")
    if any(len(a) != nsteps for a in arrs):
        logger.warning("rungs have unequal lengths; truncating all to %d steps",
                       nsteps)
    raw = np.stack([a[:nsteps] for a in arrs])       # (ntemps, nsteps, ncols)

    samples = raw[:, :, :NDIM]
    lnlike = raw[:, :, NDIM]
    lnprob = raw[:, :, NDIM + 1]
    accepted = raw[:, :, NDIM + 2]
    temps_hist = raw[:, :, NDIM + 3]
    betas = _betas_from_temperatures(temps_hist).T
    lnprior = _untemper(lnprob, betas.T, lnlike, run_dir)
    temperatures = temps_hist[:, -1]
    results = Results(
        samples=samples, lnlike=lnlike, lnprior=lnprior, lnprob=lnprob,
        accepted=accepted, betas=betas, temperatures=temperatures,
        param_names=list(PARAM_NAMES), backend="impulse",
        **_load_sidecars(run_dir, samples),
    )
    _warn_if_ladder_adapted(results, run_dir)
    return results


@register_converter("eryn")
def _convert_eryn(run_dir):
    """eryn EnsembleSampler raw output (eryn_chain.h5, group "mcmc").

    Flattened step-major into (ntemps, nsteps*nwalkers, ndim): step 0's
    walkers first, then step 1's, etc. `accepted` is each walker's
    acceptance fraction over the stored steps, broadcast per step (not a
    boolean); eryn's counter advances once per stored step, so `iteration`
    is the whole denominator regardless of thin_by.

    eryn only writes `betas` when tempering is active (ntemps > 1); untempered
    runs leave the dataset at its HDF5 fill value of 0, which would otherwise
    read back as T = inf for the cold chain, so an all-zero row is treated as
    an unwritten ladder and replaced by beta = 1. A genuine Tmax = inf top rung
    keeps its beta = 0 because the rest of its row is non-zero.

    eryn's stored log posterior is untempered, so lnprob is rebuilt as
    lnprior + beta*lnlike to match the schema. The betas used are per stored
    step, which matters under adaptive tempering: one ladder vector would be
    wrong for every step but the last.
    """
    h5py = _require_h5py()
    path = os.path.join(run_dir, "eryn_chain.h5")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{run_dir}: no eryn_chain.h5 found")

    with h5py.File(path, "r") as f:
        g = f["mcmc"]
        it = int(g.attrs["iteration"])
        if it < 1:
            raise ValueError(f"{path}: no stored iterations")
        # Datasets are over-allocated for the full run; slice to what exists.
        chain = g["chain/model_0"][:it, :, :, 0, :]  # (nsteps,ntemps,nwalkers,ndim)
        log_like = g["log_like"][:it]                # (nsteps, ntemps, nwalkers)
        log_prior = g["log_prior"][:it]
        betas = g["betas"][:it]                      # (nsteps, ntemps)
        acc_counts = g["accepted"][()]               # (ntemps, nwalkers) cumulative

    if chain.shape[-1] != NDIM:
        raise ValueError(
            f"{path}: chain has ndim={chain.shape[-1]}, expected {NDIM}")
    nsteps, ntemps, nwalkers, _ = chain.shape

    def flatten(a):  # (nsteps, ntemps, nwalkers, ...) -> (ntemps, nsteps*nwalkers, ...)
        a = np.moveaxis(a, 1, 0)
        return a.reshape(ntemps, nsteps * nwalkers, *a.shape[3:])

    if ntemps == 1 and not np.any(betas):
        logger.info("%s: no stored betas (untempered run); treating the single "
                    "rung as the beta = 1 posterior", path)
        betas = np.ones((nsteps, ntemps))

    samples = flatten(chain)
    lnlike = flatten(log_like)
    lnprior = flatten(log_prior)
    beta_draws = np.repeat(betas.T[:, :, None], nwalkers, axis=2).reshape(
        ntemps, nsteps * nwalkers)
    lnprob = _temper(lnprior, beta_draws, lnlike)
    accepted = np.tile((acc_counts / it)[:, None, :],
                       (1, nsteps, 1)).reshape(ntemps, nsteps * nwalkers)
    temperatures = _temperatures_from_betas(betas[-1])
    results = Results(
        samples=samples, lnlike=lnlike, lnprior=lnprior, lnprob=lnprob,
        accepted=accepted, betas=betas, temperatures=temperatures,
        nwalkers=nwalkers, param_names=list(PARAM_NAMES), backend="eryn",
        **_load_sidecars(run_dir, samples),
    )
    _warn_if_ladder_adapted(results, run_dir)
    return results


def main():
    ap = argparse.ArgumentParser(
        description="Convert a raw sampler output directory into the common "
                    "results.h5 format consumed by emridisp-plot.")
    ap.add_argument("run_dir", help="raw backend output directory")
    ap.add_argument("-o", "--output", default=None,
                    help=f"output path (default: <run_dir>/{DEFAULT_NAME})")
    ap.add_argument("--backend", default=None,
                    help="converter to use (default: auto-detect from "
                         "run_summary.json)")
    ap.add_argument("-f", "--force", action="store_true",
                    help="overwrite an existing results file")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out = args.output or os.path.join(args.run_dir, DEFAULT_NAME)
    if os.path.exists(out) and not args.force:
        ap.error(f"{out} exists; pass -f/--force to overwrite")

    results = convert(args.run_dir, backend=args.backend)
    results.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
