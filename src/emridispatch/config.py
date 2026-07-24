"""Load a YAML config into attribute-accessible sections.

Everything is configured through the YAML file (no environment variables). The
config path is a required CLI argument (`emridispatch <config.yaml>`); there is no
repo-root default.
"""

from types import SimpleNamespace

import yaml

from emridispatch.parameters import DEFAULT_PERIODIC_2PI_INDICES


def _as_mode(v):
    """Normalize a reparam mode: YAML parses bare `off` as boolean False."""
    if isinstance(v, bool):
        return "off" if v is False else "on"
    return str(v)


def _as_float(v):
    """Coerce injection scalars to float (YAML may leave `1.0e6` a string)."""
    return float(v)


def _merge_ns(defaults, override, **nested):
    """SimpleNamespace from defaults <- raw-dict override, plus nested sections."""
    return SimpleNamespace(**{**defaults, **(override or {})}, **nested)


def load_config(path):
    """Return a SimpleNamespace: .injection (dict) plus .data/.sampler/.prior/
    .priors/.reparam/.run/.logging/.pp sections. Backend-specific sampler knobs
    live in nested subsections: .sampler.impulse (with .ladder/.mode_jump) and
    .sampler.eryn. Defaults for every optional key are merged here, so
    consumers read attributes directly."""
    with open(path) as fh:
        raw = yaml.safe_load(fh)

    # Defensive coercion: YAML turns a bare `off` into False and an unsigned
    # exponent like `1.0e6` into a string. Normalize the mode string and the
    # float injection scalars so a config typo can't reach the sampler.
    raw["reparam"]["mode"] = _as_mode(raw["reparam"]["mode"])
    raw["injection"] = {k: _as_float(v) for k, v in raw["injection"].items()}

    raw_sampler = dict(raw.get("sampler") or {})
    raw_impulse = dict(raw_sampler.pop("impulse", None) or {})
    raw_eryn = raw_sampler.pop("eryn", None) or {}

    sampler = _merge_ns(
        {"backend": "impulse", "nsamples": 10000,
         "start_mode": "truth", "start_jitter": 5.0},
        raw_sampler,
        impulse=_merge_ns(
            {"threads": 1, "cov_update": 200, "save_freq": 200},
            {k: v for k, v in raw_impulse.items()
             if k not in ("ladder", "mode_jump")},
            ladder=_merge_ns(
                {"max_temp": 1000.0, "t_split": 25.0, "ntemps_low": 20,
                 "ntemps_high": 6, "adapt": False, "adapt_nu": 10.0,
                 "adapt_t0": 100.0},
                raw_impulse.get("ladder")),
            mode_jump=_merge_ns({"method": "none", "weight": 25.0},
                                raw_impulse.get("mode_jump"))),
        eryn=_merge_ns(
            {"nwalkers": 32, "ntemps": 1, "Tmax": None,
             "adaptive_temps": True, "adaptation_lag": 10000,
             "adaptation_time": 100, "stop_adaptation": -1, "burn": 0,
             "thin_by": 1, "progress": False, "start_spread": 1.0,
             "move": "stretch"},
            raw_eryn))

    return SimpleNamespace(
        injection=raw["injection"],                     # kept a dict for the generator
        data=SimpleNamespace(**raw["data"]),
        sampler=sampler,
        prior=_merge_ns(
            {"box_scale": 3.0, "fisher": "auto", "angle_sigma": 0.05,
             "fisher_use_gpu": None,
             "periodic_2pi_indices": list(DEFAULT_PERIODIC_2PI_INDICES)},
            raw.get("prior")),
        priors=dict(raw.get("priors", {})),             # per-parameter overrides
        reparam=SimpleNamespace(**raw["reparam"]),
        run=SimpleNamespace(**raw["run"]),
        logging=_merge_ns({"level": "INFO", "file": "run.log"},
                          raw.get("logging")),
        pp=SimpleNamespace(**raw.get("pp", {})),
    )
