"""Injection and template diagnostic plots for the workbench.

`show` selects "template"/"injection"/"data"/"noise" traces; all
functions return (fig, axes), write no files, and pass **kwargs to
the matplotlib artist. Needs matplotlib: pip install emridispatch[viz]
"""

from __future__ import annotations

import logging

import numpy as np

from emridispatch.workbench import (
    _arr, _as_template, _f_arr, _host, _require, injection_template, noise,
    to_physical)

logger = logging.getLogger(__name__)

TRACES = ("template", "injection", "data", "noise")


def _require_plotting():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "workbench plots need matplotlib; install with "
            "`pip install emridispatch[viz]`") from exc
    return plt


def _check_traces(model, show):
    names = tuple(show)
    for name in names:
        if name not in TRACES:
            raise ValueError(
                f"unknown trace {name!r}; choose from {list(TRACES)}")
    if not names:
        raise ValueError(f"show must name at least one of {list(TRACES)}")
    if "data" in names and not getattr(model, "add_noise", False):
        logger.warning(
            "data.add_noise is false; the data trace equals the noiseless "
            "injection")
    return names


def _physical(model, params):
    if params is None:
        return dict(model.injection_parameters)
    if isinstance(params, dict):
        return params
    return to_physical(model, params)


def _template_spectrum(model, params):
    """Frequency-domain template for `params`; None takes the cached injection."""
    if params is None:
        return _arr(injection_template(model))
    return _arr(_as_template(model, _physical(model, params)))


def _spectra(model, params, show):
    """{trace name: (nchannels, nf) frequency-domain array} for the shown traces."""
    out = {}
    if "template" in show:
        out["template"] = _template_spectrum(model, params)
    if "injection" in show:
        out["injection"] = _arr(injection_template(model))
    if "data" in show:
        out["data"] = _arr(model.data_residual_array)
    if "noise" in show:
        out["noise"] = noise(model)
    return {name: out[name] for name in show}


def _time_domain_length(model):
    """Padded time-domain sample count the frequency-domain data was built at."""
    data = model.data_residual_array
    settings = getattr(data, "init_kwargs", {}).get("input_signal_domain")
    n = getattr(settings, "N", None)
    if n is None:
        raise TypeError(
            "workbench_plots needs the time-domain length; "
            "model.data_residual_array exposes no "
            "init_kwargs['input_signal_domain'].N")
    return int(n)


def _to_time_domain(model, spectrum, n):
    """Strain-amplitude reconstruction of a spectrum, trimmed to `n` samples.

    Inverted on the padded grid, matching lisatools' `irfft(arr) / dt`; an irfft
    at the shorter native length would mis-space the bins.
    """
    full = np.fft.irfft(spectrum, n=_time_domain_length(model), axis=-1)
    return full[..., :n] / model.delta_t


def _time_series(model, params, show, fn):
    """{trace name: (nchannels, N) time-domain array} for the shown traces."""
    out = {}
    times = None
    if "template" in show or "injection" in show:
        _require(model, "generate_time_domain", fn)
    if "template" in show:
        times, strain = model.generate_time_domain(_physical(model, params))
        out["template"] = strain
    if "injection" in show:
        if params is None and "template" in out:
            out["injection"] = out["template"]
        else:
            times_inj, strain = model.generate_time_domain(
                dict(model.injection_parameters))
            times = times_inj if times is None else times
            out["injection"] = strain
    if times is None:
        n = _time_domain_length(model)
        times = np.arange(n) * model.delta_t
    else:
        n = next(iter(out.values())).shape[-1]
    for name in ("data", "noise"):
        if name in show:
            spectrum = (_arr(model.data_residual_array)
                        if name == "data" else noise(model))
            out[name] = _to_time_domain(model, spectrum, n)
    return times, {name: out[name] for name in show}


def _channels(model, channel, fn):
    _require(model, "channel_list", fn)
    names = list(model.channel_list)
    if channel is None:
        return names, list(range(len(names)))
    if channel not in names:
        raise ValueError(f"unknown channel {channel!r}; choose from {names}")
    return [channel], [names.index(channel)]


def _axes(plt, ax, n):
    if ax is not None:
        axes = np.atleast_1d(ax)
        return axes[0].get_figure(), axes
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(8.0, 2.6 * n))
    return fig, np.atleast_1d(axes)


def _notch_bands(model, show_notch):
    if not show_notch:
        return []
    mask = getattr(model, "_psd_notch_mask", None)
    if mask is None:
        return []
    mask = np.asarray(_host(mask), dtype=bool)
    if not mask.any():
        return []
    f = _f_arr(model.data_residual_array)
    idx = np.flatnonzero(mask)
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return [(f[g[0]], f[g[-1]]) for g in groups]


def _shade(axis, bands):
    for lo, hi in bands:
        axis.axvspan(lo, hi, color="0.8", alpha=0.5, zorder=0)


def _default_nperseg(n):
    """STFT window for an `n`-sample series: 2**floor(log2(sqrt(n)))."""
    n = int(n)
    if n < 8:
        return n
    return int(min(n, max(8, 2 ** int(np.floor(np.log2(np.sqrt(n)))))))


def _log_norm(amplitude, dynamic_range):
    """(norm, clipped amplitude, floor) for `dynamic_range` dB below the peak.

    Norm and floor are None when nothing is positive.
    """
    from matplotlib.colors import LogNorm

    peak = float(np.nanmax(amplitude)) if amplitude.size else 0.0
    if not np.isfinite(peak) or peak <= 0.0:
        return None, amplitude, None
    floor = peak * 10.0 ** (-float(dynamic_range) / 20.0)
    return LogNorm(vmin=floor, vmax=peak), np.maximum(amplitude, floor), floor


def _auto_flim(f, amplitude, floor):
    """Frequency limits framing the bins that rise above `floor`, padded 2x."""
    if floor is None:
        return f[0], f[-1]
    band = f[np.nanmax(amplitude, axis=-1) > floor]
    if not band.size:
        return f[0], f[-1]
    return max(band.min() / 2.0, f[0]), min(band.max() * 2.0, f[-1])


def _auto_tlim(t, amplitude, floor):
    """Time limits framing the segments that rise above `floor`, padded 2%."""
    if floor is None:
        return t[0], t[-1]
    span = t[np.nanmax(amplitude, axis=0) > floor]
    if not span.size:
        return t[0], t[-1]
    pad = 0.02 * float(span.max() - span.min())
    return max(span.min() - pad, t[0]), min(span.max() + pad, t[-1])


def plot_time_frequency(model, params=None, *, show=("template",), channel=None,
                        dynamic_range=60.0, flim=None, tlim=None,
                        colorbar=True, stft_kwargs=None, ax=None, **kwargs):
    """Spectrogram of each shown trace, one panel per channel per trace.

    Colour is log |STFT|, `dynamic_range` dB below each panel's peak
    (f=0 dropped); `flim`/`tlim` default to the band/span above floor,
    else pass (lo, hi); `dynamic_range=None` or norm/vmin/vmax disables it.
    """
    from scipy.signal import stft

    plt = _require_plotting()
    names = _check_traces(model, show)
    chan_names, indices = _channels(model, channel, "plot_time_frequency")
    times, series = _time_series(model, params, names, "plot_time_frequency")
    fs = 1.0 / float(times[1] - times[0])
    first = next(iter(series.values()))
    opts = dict(nperseg=_default_nperseg(first.shape[-1]))
    opts.update(stft_kwargs or {})
    scaled_by_caller = bool({"norm", "vmin", "vmax"} & set(kwargs))

    fig, axes = _axes(plt, ax, len(chan_names) * len(names))
    panel = 0
    spans = []
    for trace in names:
        strain = series[trace]
        for chan_name, i in zip(chan_names, indices):
            f, t, Z = stft(strain[i], fs=fs, **opts)
            f, amplitude = f[1:], np.abs(Z[1:])
            mesh_kwargs = dict(kwargs)
            floor = None
            if dynamic_range is not None and not scaled_by_caller:
                norm, amplitude, floor = _log_norm(amplitude, dynamic_range)
                if norm is not None:
                    mesh_kwargs["norm"] = norm
            mesh = axes[panel].pcolormesh(t, f, amplitude, shading="auto",
                                          **mesh_kwargs)
            axes[panel].set_yscale("log")
            axes[panel].set_ylim(*(flim if flim is not None
                                   else _auto_flim(f, amplitude, floor)))
            axes[panel].set_ylabel(f"{trace} {chan_name}  f [Hz]")
            spans.append(_auto_tlim(t, amplitude, floor))
            if colorbar:
                fig.colorbar(mesh, ax=axes[panel], label="|STFT|")
            panel += 1
    span = tlim if tlim is not None else (min(lo for lo, _ in spans),
                                          max(hi for _, hi in spans))
    for axis in axes:
        axis.set_xlim(*span)
    axes[-1].set_xlabel("t [s]")
    return fig, axes


def plot_char_strain(model, params=None, *, show=("template",), show_notch=True,
                     ax=None, **kwargs):
    """f|h(f)| against the sensitivity curve, per channel."""
    plt = _require_plotting()
    _require(model, "sensitivity_matrix", "plot_char_strain")
    _require(model, "channel_list", "plot_char_strain")
    names = _check_traces(model, show)
    spectra = _spectra(model, params, names)
    f = _f_arr(model.data_residual_array)
    sens = np.asarray(_host(model.sensitivity_matrix.sens_mat))
    bands = _notch_bands(model, show_notch)

    fig, axes = _axes(plt, ax, len(model.channel_list))
    for i, chan_name in enumerate(model.channel_list):
        _shade(axes[i], bands)
        for trace in names:
            axes[i].loglog(f, f * np.abs(spectra[trace][i]), label=trace,
                           **kwargs)
        axes[i].loglog(f, np.sqrt(f * sens[i]), label="sensitivity",
                       color="0.4", linestyle="--")
        axes[i].set_ylabel(f"{chan_name}  char. strain")
        axes[i].legend(loc="best", fontsize="small")
    axes[-1].set_xlabel("f [Hz]")
    return fig, axes


def plot_snr_accumulation(model, params=None, *, show=("template",),
                          show_notch=True, ax=None, **kwargs):
    """Cumulative SNR against frequency, summed over channels.

    "template"/"injection" accumulate optimal SNR and are monotonic; "data"/
    "noise" accumulate the detected statistic and can decrease or go negative.
    """
    plt = _require_plotting()
    _require(model, "sensitivity_matrix", "plot_snr_accumulation")
    names = _check_traces(model, show)
    inv = np.asarray(_host(model.sensitivity_matrix.invC))
    differential = model.sensitivity_matrix.differential_component
    f = _f_arr(model.data_residual_array)

    h = None
    norm = 1.0
    if set(names) & {"template", "data", "noise"}:
        h = _template_spectrum(model, params)
        h_h_total = 0.0
        for i in range(h.shape[0]):
            row = np.real(h[i].conj() * h[i]) * inv[i]
            h_h_total += 4.0 * float(np.nansum(row)) * differential
        if h_h_total > 0.0:
            norm = np.sqrt(h_h_total)

    spectra = _spectra(model, params, [n for n in names if n != "template"])
    if "template" in names:
        spectra["template"] = h
    spectra = {name: spectra[name] for name in names}

    fig, axes = _axes(plt, ax, 1)
    _shade(axes[0], _notch_bands(model, show_notch))
    for trace in names:
        arr = spectra[trace]
        per_bin = np.zeros(f.shape[-1])
        for i in range(arr.shape[0]):
            if trace in ("template", "injection"):
                row = np.real(arr[i].conj() * arr[i]) * inv[i]
            else:
                row = np.real(arr[i].conj() * h[i]) * inv[i]
            per_bin = per_bin + np.nan_to_num(
                row, nan=0.0, posinf=0.0, neginf=0.0)
        cumulative = np.cumsum(4.0 * per_bin * differential)
        curve = (np.sqrt(cumulative) if trace in ("template", "injection")
                 else cumulative / norm)
        axes[0].plot(f, curve, label=trace, **kwargs)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("f [Hz]")
    axes[0].set_ylabel("cumulative SNR")
    axes[0].legend(loc="best", fontsize="small")
    return fig, axes


def plot_time_domain(model, params=None, *, show=("template",), residual=False,
                     whiten=False, ax=None, **kwargs):
    """h(t) per channel; residual=True plots d - h(params), not d - h(injection)."""
    plt = _require_plotting()
    _require(model, "channel_list", "plot_time_domain")
    names = _check_traces(model, show)
    times, series = _time_series(model, params, names, "plot_time_domain")

    if residual and "template" in series:
        n = series["template"].shape[-1]
        data_td = _to_time_domain(model, _arr(model.data_residual_array), n)
        series = dict(series)
        series["template"] = data_td - series["template"]

    if whiten:
        _require(model, "sensitivity_matrix", "plot_time_domain")
        sens = np.asarray(_host(model.sensitivity_matrix.sens_mat))
        scale = np.sqrt(np.where(np.isfinite(sens) & (sens > 0), sens, np.inf))
        whitened = {}
        for trace, strain in series.items():
            spec = np.fft.rfft(strain, axis=-1)
            m = min(spec.shape[-1], scale.shape[-1])
            spec[:, :m] = spec[:, :m] / scale[:, :m]
            spec[:, m:] = 0.0
            whitened[trace] = np.fft.irfft(spec, n=strain.shape[-1], axis=-1)
        series = whitened

    fig, axes = _axes(plt, ax, len(model.channel_list))
    for i, chan_name in enumerate(model.channel_list):
        for trace in names:
            label = "d - h" if (residual and trace == "template") else trace
            axes[i].plot(times, series[trace][i], label=label, **kwargs)
        axes[i].set_ylabel(chan_name)
        axes[i].legend(loc="best", fontsize="small")
    axes[-1].set_xlabel("t [s]")
    return fig, axes
