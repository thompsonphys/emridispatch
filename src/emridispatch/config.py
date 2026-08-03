"""Load a YAML config into attribute-accessible sections.

No environment variables; the config path is a required CLI argument
(`emridispatch <config.yaml>`), with no repo-root default.
"""

from types import SimpleNamespace

import yaml

from emridispatch.parameters import DEFAULT_PERIODIC_2PI_INDICES


def _as_mode(v):
    """Normalize a reparam mode: YAML parses bare `off` as boolean False."""
    if isinstance(v, bool):
        return "off" if v is False else "on"
    return str(v)


_TDI_ALIASES = {
    "off": "off",
    "1st": "1st generation", "1st generation": "1st generation",
    "2nd": "2nd generation", "2nd generation": "2nd generation",
}


def _as_tdi(v):
    if isinstance(v, bool):
        return "2nd generation" if v else "off"
    s = str(v).strip().lower()
    if s not in _TDI_ALIASES:
        raise ValueError(
            f"data.tdi must be one of off | 1st | 2nd "
            f"(or '1st generation' / '2nd generation') (got {v!r})")
    return _TDI_ALIASES[s]


def _merge_ns(defaults, override, **nested):
    """SimpleNamespace from defaults <- raw-dict override, plus nested sections."""
    return SimpleNamespace(**{**defaults, **(override or {})}, **nested)


DATA_DEFAULTS = {
    "response": "lisatools",
    "duration": 1.0,
    "delta_t": 10.0,
    "inj_snr": None,
    "channels": None,
    "tdi": "2nd generation",
    "foreground": True,
    "add_noise": False,
    "noise_seed": 0,
    "pad_fft": True,
    "psd_notch": 1.0e-5,
    "psd_notch_depth": 2.0,
    "psd_notch_strict": True,
    "toy_sigma_scale": 1.0,
}


SAMPLER_DEFAULTS = {
    "backend": "impulse", "nsamples": 10000,
    "start_mode": "truth", "start_jitter": 5.0,
}
IMPULSE_DEFAULTS = {"threads": 1, "cov_update": 200, "save_freq": 200}
LADDER_DEFAULTS = {
    "max_temp": 1000.0, "t_split": 25.0, "ntemps_low": 20,
    "ntemps_high": 6, "adapt": False, "adapt_nu": 10.0, "adapt_t0": 100.0,
}
MODE_JUMP_DEFAULTS = {"method": "none", "weight": 25.0}
ERYN_DEFAULTS = {
    "nwalkers": 32, "ntemps": 1, "Tmax": None,
    "adaptive_temps": True, "adaptation_lag": 10000,
    "adaptation_time": 100, "stop_adaptation": -1, "burn": 0,
    "thin_by": 1, "progress": False, "start_spread": 1.0, "move": "stretch",
}
PRIOR_DEFAULTS = {
    "box_scale": 3.0, "fisher": "auto", "angle_sigma": 0.05,
    "fisher_use_gpu": None, "periodic_2pi_indices": None,
    "sigmas": None, "covariance_file": None,
}
INJECTION_KEYS = (
    "mass_1", "mass_2", "a", "p", "e", "x", "luminosity_distance",
    "q_s", "phi_s", "q_k", "phi_k", "phi_phi", "phi_theta", "phi_r",
)
REPARAM_KEYS = ("mode", "idx")
RUN_KEYS = ("outdir", "seed")
LOGGING_DEFAULTS = {"level": "INFO", "file": "run.log"}
PP_KEYS = ("nruns", "outroot", "draw_seed", "nsamples", "burn_frac")

# Sections with no defaults to fall back on, and the keys within them that are
# dereferenced unconditionally.
REQUIRED_SECTIONS = {
    "injection": INJECTION_KEYS,
    "data": (),
    "reparam": REPARAM_KEYS,
    "run": RUN_KEYS,
}
TOP_LEVEL_SECTIONS = (
    "injection", "data", "sampler", "prior", "priors", "reparam", "run",
    "logging", "pp",
)


def _check_keys(section, raw, allowed):
    unknown = sorted(set(raw or ()) - set(allowed))
    if unknown:
        import difflib

        hints = []
        for key in unknown:
            near = difflib.get_close_matches(key, allowed, n=1, cutoff=0.7)
            hints.append(f"{key!r}" + (f" (did you mean {near[0]!r}?)" if near else ""))
        raise ValueError(
            f"unknown key(s) in config section {section!r}: {', '.join(hints)}. "
            f"Valid keys: {', '.join(sorted(allowed))}")


def _check_required(raw):
    absent = [s for s in REQUIRED_SECTIONS if not isinstance(raw.get(s), dict)]
    if absent:
        raise ValueError(
            f"missing config section(s): {', '.join(absent)}. Required: "
            f"{', '.join(REQUIRED_SECTIONS)}")
    for section, keys in REQUIRED_SECTIONS.items():
        missing = sorted(set(keys) - set(raw[section]))
        if missing:
            raise ValueError(
                f"missing key(s) in config section {section!r}: "
                f"{', '.join(missing)}")


def load_config(path):
    """Return a SimpleNamespace of config sections parsed from YAML.

    Sections: injection (dict) plus .data/.sampler/.prior/.priors/
    .reparam/.run/.logging/.pp. Sampler knobs nest under .sampler.impulse
    (.ladder/.mode_jump) and .sampler.eryn, each with merged-in defaults.
    """
    with open(path) as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"config {path} is empty or not a YAML mapping")
    _check_keys("<top level>", raw, TOP_LEVEL_SECTIONS)
    _check_required(raw)

    # Defensive coercion: YAML turns a bare `off` into False and an unsigned
    # exponent like `1.0e6` into a string. Normalize the mode string and the
    # float injection scalars so a config typo can't reach the sampler.
    raw["reparam"]["mode"] = _as_mode(raw["reparam"]["mode"])
    _check_keys("injection", raw["injection"], INJECTION_KEYS)
    injection = {}
    for k, v in raw["injection"].items():
        try:
            injection[k] = float(v)
        except (TypeError, ValueError):
            raise ValueError(
                f"injection.{k} must be a number (got {v!r})") from None
    raw["injection"] = injection

    raw_data = dict(raw["data"])
    _check_keys("data", raw_data, DATA_DEFAULTS)
    raw_data["tdi"] = _as_tdi(raw_data.get("tdi", "2nd generation"))
    raw_data["foreground"] = bool(raw_data.get("foreground", True))

    raw_sampler = dict(raw.get("sampler") or {})
    raw_impulse = dict(raw_sampler.pop("impulse", None) or {})
    raw_eryn = raw_sampler.pop("eryn", None) or {}
    raw_ladder = raw_impulse.get("ladder") or {}
    raw_mode_jump = raw_impulse.get("mode_jump") or {}

    _check_keys("sampler", raw_sampler, SAMPLER_DEFAULTS)
    _check_keys("sampler.impulse",
                {k: v for k, v in raw_impulse.items()
                 if k not in ("ladder", "mode_jump")}, IMPULSE_DEFAULTS)
    _check_keys("sampler.impulse.ladder", raw_ladder, LADDER_DEFAULTS)
    _check_keys("sampler.impulse.mode_jump", raw_mode_jump, MODE_JUMP_DEFAULTS)
    _check_keys("sampler.eryn", raw_eryn, ERYN_DEFAULTS)
    _check_keys("prior", raw.get("prior"), PRIOR_DEFAULTS)
    _check_keys("reparam", raw.get("reparam"), REPARAM_KEYS)
    _check_keys("run", raw.get("run"), RUN_KEYS)
    _check_keys("logging", raw.get("logging"), LOGGING_DEFAULTS)
    _check_keys("pp", raw.get("pp"), PP_KEYS)

    sampler = _merge_ns(
        SAMPLER_DEFAULTS,
        raw_sampler,
        impulse=_merge_ns(
            IMPULSE_DEFAULTS,
            {k: v for k, v in raw_impulse.items()
             if k not in ("ladder", "mode_jump")},
            ladder=_merge_ns(LADDER_DEFAULTS, raw_ladder),
            mode_jump=_merge_ns(MODE_JUMP_DEFAULTS, raw_mode_jump)),
        eryn=_merge_ns(ERYN_DEFAULTS, raw_eryn))

    return SimpleNamespace(
        injection=raw["injection"],                     # kept a dict for the generator
        data=_merge_ns(DATA_DEFAULTS, raw_data),
        sampler=sampler,
        prior=_merge_ns(
            dict(PRIOR_DEFAULTS,
                 periodic_2pi_indices=list(DEFAULT_PERIODIC_2PI_INDICES)),
            raw.get("prior")),
        priors=dict(raw.get("priors") or {}),           # per-parameter overrides
        reparam=SimpleNamespace(**raw["reparam"]),
        run=SimpleNamespace(**raw["run"]),
        logging=_merge_ns(LOGGING_DEFAULTS, raw.get("logging")),
        pp=SimpleNamespace(**(raw.get("pp") or {})),
    )
