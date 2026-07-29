"""Sampler-agnostic results: raw backend output -> common HDF5 format.

Converts a backend outdir (impulse chain_N.txt; eryn eryn_chain.h5)
into one format_version 1 results.h5. Requires h5py
(pip install emridispatch[results]).

Schema; root attrs format_version, backend, ndim, ntemps, nsteps,
param_names (json), config (json):
    /chains/samples    (ntemps, nsteps, ndim)  raw sampling coords
    /chains/lnlike     (ntemps, nsteps)
    /chains/lnprob     (ntemps, nsteps)
    /chains/accepted   (ntemps, nsteps)
    /temperatures      (ntemps,)               inf allowed (top rung)
    /physical/samples  (ntemps, nsteps, ndim)  whitening inverted
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

    samples: np.ndarray            # (ntemps, nsteps, ndim) sampling coords
    lnlike: np.ndarray             # (ntemps, nsteps)
    lnprob: np.ndarray             # (ntemps, nsteps)
    accepted: np.ndarray           # (ntemps, nsteps)
    temperatures: np.ndarray       # (ntemps,)
    param_names: list = field(default_factory=lambda: list(PARAM_NAMES))
    physical: np.ndarray | None = None   # (ntemps, nsteps, ndim) or None
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

    # --- structure -----------------------------------------------------------
    @property
    def ntemps(self):
        return self.samples.shape[0]

    @property
    def nsteps(self):
        return self.samples.shape[1]

    @property
    def ndim(self):
        return self.samples.shape[2]

    # --- accessors -----------------------------------------------------------
    def rung(self, i, physical=True, burn=0, thin=1):
        """(nkept, ndim) samples of one rung. thin is cosmetic (plot rendering),
        not a statistical device -- quote MC error via ESS instead."""
        data = self.physical if physical else self.samples
        if data is None:
            raise ValueError(
                "no physical-coordinate samples stored (conversion had no "
                "reparam transform); use physical=False")
        s = data[i, burn:][::thin]
        if len(s) <= 1:
            raise ValueError(
                f"after burn={burn} only {len(s)} draws remain; reduce --burn")
        return s

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
            f.attrs["nsteps"] = self.nsteps
            f.attrs["param_names"] = json.dumps(list(self.param_names))
            f.attrs["config"] = json.dumps(self.config)

            g = f.create_group("chains")
            g.create_dataset("samples", data=self.samples)
            g.create_dataset("lnlike", data=self.lnlike)
            g.create_dataset("lnprob", data=self.lnprob)
            g.create_dataset("accepted", data=self.accepted)
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
                    f"(expected {FORMAT_VERSION})")
            kw = dict(
                samples=f["chains/samples"][()],
                lnlike=f["chains/lnlike"][()],
                lnprob=f["chains/lnprob"][()],
                accepted=f["chains/accepted"][()],
                temperatures=f["temperatures"][()],
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


@register_converter("impulse")
def _convert_impulse(run_dir):
    """impulse PTSampler raw output: chain_N.txt per rung, cols
    [params(ndim), lnlike, lnprob, accepted, temperature]."""
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
    temperatures = raw[:, 0, NDIM + 3]

    return Results(
        samples=samples, lnlike=lnlike, lnprob=lnprob, accepted=accepted,
        temperatures=temperatures, param_names=list(PARAM_NAMES),
        backend="impulse", **_load_sidecars(run_dir, samples),
    )


@register_converter("eryn")
def _convert_eryn(run_dir):
    """eryn EnsembleSampler raw output (eryn_chain.h5, group "mcmc").

    Flattened STEP-MAJOR into (ntemps, nsteps*nwalkers, ndim): step 0's
    walkers first, then step 1's, etc. `accepted` is each walker's
    acceptance fraction over the stored steps, broadcast per step (not a
    boolean); eryn's counter advances once per stored step, so `iteration`
    is the whole denominator regardless of thin_by.

    Temperatures come from the last stored beta row. eryn only writes
    `betas` when tempering is active (ntemps > 1); untempered runs leave
    the dataset at its HDF5 fill value of 0, which would otherwise read
    back as T = inf for the cold chain, so an all-zero row is treated as
    an unwritten ladder and mapped to T = 1. A genuine Tmax = inf top rung
    keeps its beta = 0 because the rest of its row is non-zero.
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

    samples = flatten(chain)
    lnlike = flatten(log_like)
    lnprob = flatten(log_like + log_prior)
    accepted = np.tile((acc_counts / it)[:, None, :],
                       (1, nsteps, 1)).reshape(ntemps, nsteps * nwalkers)
    if np.any(betas[-1]):
        with np.errstate(divide="ignore"):
            temperatures = np.where(betas[-1] > 0.0, 1.0 / betas[-1], np.inf)
    else:
        logger.info("eryn_chain.h5 has no stored betas (untempered run); "
                    "labelling the single rung T = 1")
        temperatures = np.ones(ntemps)

    return Results(
        samples=samples, lnlike=lnlike, lnprob=lnprob, accepted=accepted,
        temperatures=temperatures, param_names=list(PARAM_NAMES),
        backend="eryn", **_load_sidecars(run_dir, samples),
    )


def main():
    ap = argparse.ArgumentParser(
        description="Convert a raw sampler output directory into the common "
                    "results.h5 format consumed by emridispatch-plot.")
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
