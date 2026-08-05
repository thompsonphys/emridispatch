"""Interactive exploration of an emridispatch injection: SNR, likelihood, priors.

Read-only: no writes to the run directory, no Fisher unless allowed.
Plots live in emridispatch.workbench_plots (needs the `viz` extra).
"""

import inspect
import logging
from dataclasses import dataclass

import numpy as np

from emridispatch.parameters import (
    NDIM, PARAM_NAMES, VECTOR_TO_PHYSICAL, physical_from_vector, truth_vector)

logger = logging.getLogger(__name__)

__all__ = ["load", "truth", "to_vector", "to_physical", "offset", "signal", "lnlike",
           "SNR", "snr", "overlap", "Measurement", "measure", "injection_template", "noise",
           "prior_from_config"]

_SAMPLED = frozenset(VECTOR_TO_PHYSICAL.values())


def truth(model):
    """The 12-D sampling vector at the model's injection."""
    return truth_vector(model.injection_parameters)


def to_vector(params):
    """Coerce a physical-parameter dict or array-like to a 12-D sampling vector."""
    if isinstance(params, dict):
        return truth_vector(params)
    vec = np.asarray(params, dtype=float)
    if vec.shape != (NDIM,):
        raise ValueError(
            f"sampling vector must have shape ({NDIM},), got {vec.shape}")
    return vec


def to_physical(model, vector):
    """Physical-parameter dict for a 12-D sampling vector.

    Injection keys the vector does not carry are copied from
    model.injection_parameters: the equatorial waveform never samples them.
    """
    fiducial = {name: float(val)
                for name, val in model.injection_parameters.items()
                if name not in _SAMPLED}
    return physical_from_vector(to_vector(vector), fiducial)


def offset(model, **deltas):
    """Truth vector with additive per-parameter deltas in sampling coordinates.

    Keys are PARAM_NAMES, so `offset(model, ln_m1=0.01)` is a log-mass offset
    while `offset(model, p=0.05)` is linear in the semi-latus rectum.
    """
    vec = truth(model).copy()
    for name, delta in deltas.items():
        if name not in PARAM_NAMES:
            raise ValueError(
                f"unknown parameter {name!r}; choose from {PARAM_NAMES}")
        vec[PARAM_NAMES.index(name)] += float(delta)
    return vec


def load(config_path):
    """(cfg, model) for a config file: no outdir, no logging, no writes.

    set_env_guards() runs, but its BLAS/OMP thread vars only take effect
    before numpy's first import; call it yourself first in scripts.
    """
    from emridispatch.cli import set_env_guards
    from emridispatch.config import load_config
    from emridispatch.response import build_injection_model

    set_env_guards()
    cfg = load_config(config_path)
    return cfg, build_injection_model(cfg)


_CAPABILITY = {
    "generate_signal": "signal generator",
    "analysis_container": "analysis container",
    "sensitivity_matrix": "sensitivity matrix",
    "generate_time_domain": "time-domain generator",
    "channel_list": "channel list",
}


def _require(model, attr, fn):
    """Raise unless the model provides attr; names the config knob to change."""
    if getattr(model, attr, None) is None:
        response = type(model).__name__
        raise TypeError(
            f"workbench.{fn} needs a waveform model; response {response!r} has "
            f"no {_CAPABILITY[attr]}")


def _host(value):
    """Bring a cupy scalar/array back to the host; pass numpy through."""
    return value.get() if hasattr(value, "get") else value


def _arr(container):
    """Host-side channel array for a data/template container."""
    inner = getattr(container, "data_res_arr", container)
    return np.asarray(_host(inner.arr))


def _f_arr(container):
    """Host-side frequency grid for a data/template container.

    The wrapper's own f_arr always raises; read from the inner object.
    """
    inner = getattr(container, "data_res_arr", container)
    return np.asarray(_host(inner.f_arr))


def _is_template(target):
    return not isinstance(target, dict) and not hasattr(target, "__len__")


def signal(model, params):
    """Template for a parameter set, reusable across measures."""
    _require(model, "generate_signal", "signal")
    payload = params if isinstance(params, dict) else to_physical(model, params)
    return model.generate_signal(payload)


def _evaluate(model, payload, full):
    """model.evaluate_likelihood, tolerating models without a full= kwarg."""
    if "full" in inspect.signature(model.evaluate_likelihood).parameters:
        return model.evaluate_likelihood(payload, full=full)
    if full:
        raise TypeError(
            f"workbench.lnlike full=True needs a model whose "
            f"evaluate_likelihood accepts full=; "
            f"{type(model).__name__} does not")
    return model.evaluate_likelihood(payload)


def lnlike(model, target, full=False):
    """ln L for a parameter set or a prebuilt template.

    full=False (default) is the template-varying part <d|h> - <h|h>/2, matching
    the sampler's convention; full=True is the normalised absolute ln L.
    """
    if isinstance(target, dict):
        payload = target
    elif _is_template(target):
        _require(model, "generate_signal", "lnlike")
        payload = target
    else:
        payload = to_physical(model, target)
    return float(np.real(_host(_evaluate(model, payload, full))))


@dataclass
class SNR:
    """Optimal sqrt(<h|h>) and detected <d|h>/sqrt(<h|h>) signal-to-noise."""

    optimal: float
    detected: float
    per_channel: dict | None = None


def _as_template(model, target):
    if isinstance(target, dict):
        return signal(model, target)
    if _is_template(target):
        return target
    return signal(model, to_physical(model, target))


def _channel_inner(model, a, b):
    """Per-channel <a|b>, reproducing diagnostic.inner_product for a diagonal PSD."""
    sens = model.sensitivity_matrix
    inv = _host(sens.invC)
    differential = sens.differential_component
    arr_a, arr_b = _arr(a), _arr(b)
    out = []
    for i in range(arr_a.shape[0]):
        start = 1 if np.isnan(inv[i][0]) else 0
        y = np.real(arr_a[i][start:].conj() * arr_b[i][start:]) * inv[i][start:]
        out.append(4.0 * float(np.sum(y)) * differential)
    return np.asarray(out)


def snr(model, target, phase_maximize=False, per_channel=False):
    """Optimal and detected SNR for a parameter set or prebuilt template."""
    if phase_maximize and per_channel:
        raise ValueError(
            "snr: phase_maximize is over one global phase and has no "
            "per-channel split; use one or the other")
    _require(model, "analysis_container", "snr")
    template = _as_template(model, target)
    optimal, detected = model.analysis_container.template_snr(
        template, phase_maximize=phase_maximize)
    result = SNR(float(np.real(_host(optimal))), float(np.real(_host(detected))))
    if not per_channel:
        return result

    _require(model, "sensitivity_matrix", "snr")
    _require(model, "channel_list", "snr")
    h_h = _channel_inner(model, template, template)
    d_h = _channel_inner(model, model.data_residual_array, template)
    opt = np.sqrt(h_h)
    with np.errstate(divide="ignore", invalid="ignore"):
        det = np.where(opt > 0.0, d_h / np.where(opt > 0.0, opt, 1.0), 0.0)
    result.per_channel = {
        name: SNR(float(opt[i]), float(det[i]))
        for i, name in enumerate(model.channel_list)
    }
    return result


def overlap(model, target, phase_maximize=False):
    """Normalised <d|h>/sqrt(<d|d><h|h>). Mismatch is 1 - this."""
    _require(model, "sensitivity_matrix", "overlap")
    template = _as_template(model, target)
    data = model.data_residual_array
    d_h = float(np.sum(_channel_inner(model, data, template)))
    d_d = float(np.sum(_channel_inner(model, data, data)))
    h_h = float(np.sum(_channel_inner(model, template, template)))
    norm = np.sqrt(d_d * h_h)
    if norm <= 0.0:
        return float("nan")
    value = d_h / norm
    return abs(value) if phase_maximize else value


@dataclass
class Measurement:
    """Every scalar measure for one parameter set, from one waveform generation."""

    snr: SNR
    lnlike: float
    lnlike_full: float
    overlap: float


def measure(model, params, phase_maximize=False, per_channel=False):
    """All scalar measures for a parameter set, generating the waveform once."""
    template = _as_template(model, params)
    return Measurement(
        snr=snr(model, template, phase_maximize=phase_maximize,
                per_channel=per_channel),
        lnlike=lnlike(model, template),
        lnlike_full=lnlike(model, template, full=True),
        overlap=overlap(model, template, phase_maximize=phase_maximize),
    )


_INJECTION_CACHE_ATTR = "_workbench_injection_template"


def injection_template(model):
    """The noiseless injected signal, regenerated once and cached on the model.

    Regenerated from injection_parameters since data_residual_array only
    holds the noisy version. Cached per model instance; reassigning
    injection_parameters afterward does not invalidate the cache.
    """
    _require(model, "generate_signal", "injection_template")
    cached = getattr(model, _INJECTION_CACHE_ATTR, None)
    if cached is None:
        cached = model.generate_signal(dict(model.injection_parameters))
        setattr(model, _INJECTION_CACHE_ATTR, cached)
    return cached


def noise(model):
    """The noise realization alone: d - h_injection, shape (nchannels, nf)."""
    if not getattr(model, "add_noise", False):
        logger.warning(
            "noise: data.add_noise is false; returning zeros. Set "
            "data.add_noise: true to inject a realization")
    data = _arr(model.data_residual_array)
    injected = _arr(injection_template(model))
    return data - injected


def prior_from_config(cfg, model=None, allow_fisher=False):
    """The run's JointPrior, from the outdir cache when present.

    Resolution order: prior_spec.json, then prior_bounds.npz, then a
    fresh build (never writes the cache). Fresh build only needs a model
    when data.inj_snr calibrates the injected distance and none was passed.
    """
    import json
    import os

    from emridispatch.bounds import cache_path, load_prior_bounds
    from emridispatch.priors import joint_prior_from_box, joint_prior_from_specs

    outdir = cfg.run.outdir
    spec_path = os.path.join(outdir, "prior_spec.json")
    if os.path.exists(spec_path):
        with open(spec_path) as fh:
            return joint_prior_from_specs(json.load(fh))

    reparam_idx = np.array(cfg.reparam.idx)
    box_scale = float(cfg.prior.box_scale)
    bounds_cache = cache_path(outdir)
    if os.path.exists(bounds_cache):
        from emridispatch.pipeline import fisher_key_from_config

        mins, maxes, _cov, _reparam, _mode = load_prior_bounds(
            bounds_cache, NDIM, reparam_idx, cfg.reparam.mode,
            box_scale=box_scale, fisher_key=fisher_key_from_config(cfg))
    else:
        mins, maxes = _box_from_config(cfg, reparam_idx, box_scale, allow_fisher,
                                        model)

    return joint_prior_from_box(
        mins, maxes, cfg.prior.periodic_2pi_indices, names=PARAM_NAMES,
        overrides=cfg.priors)


def _box_from_config(cfg, reparam_idx, box_scale, allow_fisher, model=None):
    from emridispatch.bounds import build_prior_bounds
    from emridispatch.fisher import get_fisher_provider

    provider = get_fisher_provider(cfg)
    if provider.name == "sef" and not allow_fisher:
        raise RuntimeError(
            "prior.fisher resolves to 'sef', which costs GPU-hours; pass "
            "allow_fisher=True, set prior.fisher to manual/none, or point "
            "run.outdir at a run with a cached prior")

    if model is not None:
        injection_parameters = model.injection_parameters
    elif cfg.data.inj_snr is None:
        injection_parameters = dict(cfg.injection)
    else:
        from emridispatch.response import build_injection_model

        injection_parameters = build_injection_model(cfg).injection_parameters

    truth_vec = truth_vector(injection_parameters)
    fisher = provider.compute(
        injection_parameters, duration=cfg.data.duration,
        delta_t=cfg.data.delta_t, use_gpu=cfg.prior.fisher_use_gpu)
    mins, maxes, _cov, _reparam = build_prior_bounds(
        fisher.sigmas, fisher.cov, fisher.order, injection_parameters,
        truth_vec, cfg.reparam.mode, reparam_idx, cfg.prior.angle_sigma, NDIM,
        box_scale=box_scale)
    return mins, maxes
