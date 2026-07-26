"""Smoke tests for the workbench plots: shapes and plumbing, not appearance."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

from workbench_stub import (
    CHANNELS, DF, N_TIME, N_TIME_NATIVE, NCHAN, NF, StubModel, _Domain)


@pytest.fixture(autouse=True)
def close_figures():
    """Discard every figure a test opened; the plot functions return, not close."""
    yield
    plt.close("all")


@pytest.fixture
def model():
    return StubModel()


def test_time_frequency_one_panel_per_channel(model):
    from emridispatch.workbench_plots import plot_time_frequency

    fig, axes = plot_time_frequency(model, {"e": 0.1})
    assert len(np.atleast_1d(axes)) == len(CHANNELS)


def test_time_frequency_single_channel(model):
    from emridispatch.workbench_plots import plot_time_frequency

    fig, axes = plot_time_frequency(model, {"e": 0.1}, channel="A")
    assert len(np.atleast_1d(axes)) == 1


def test_time_frequency_forwards_stft_kwargs(model):
    from emridispatch.workbench_plots import plot_time_frequency

    fig, axes = plot_time_frequency(
        model, {"e": 0.1}, stft_kwargs={"nperseg": 32})
    assert fig is not None


def test_char_strain_defaults_to_injection(model):
    from emridispatch.workbench_plots import plot_char_strain

    fig, axes = plot_char_strain(model)
    assert len(np.atleast_1d(axes)) == len(CHANNELS)


def test_char_strain_default_show_is_template_only(model):
    from emridispatch.workbench_plots import plot_char_strain

    fig, axes = plot_char_strain(model, {"e": 0.1})
    labels = [line.get_label() for line in np.atleast_1d(axes)[0].get_lines()]
    assert "template" in labels
    assert "data" not in labels
    assert "noise" not in labels


def test_char_strain_overlays_requested_traces(model):
    from emridispatch.workbench_plots import plot_char_strain

    fig, axes = plot_char_strain(
        model, {"e": 0.1}, show=("template", "data", "noise"))
    labels = [line.get_label() for line in np.atleast_1d(axes)[0].get_lines()]
    for name in ("template", "data", "noise"):
        assert name in labels


def test_char_strain_data_only(model):
    from emridispatch.workbench_plots import plot_char_strain

    fig, axes = plot_char_strain(model, {"e": 0.1}, show=("data",))
    labels = [line.get_label() for line in np.atleast_1d(axes)[0].get_lines()]
    assert "data" in labels
    assert "template" not in labels


def test_char_strain_shows_injection_trace(model):
    from emridispatch.workbench_plots import plot_char_strain

    fig, axes = plot_char_strain(model, {"e": 0.1},
                                 show=("template", "injection"))
    labels = [line.get_label() for line in np.atleast_1d(axes)[0].get_lines()]
    assert "injection" in labels


def test_unknown_trace_raises(model):
    from emridispatch.workbench_plots import plot_char_strain

    with pytest.raises(ValueError, match="unknown trace"):
        plot_char_strain(model, {"e": 0.1}, show=("signal",))


def test_data_trace_warns_when_add_noise_off(model, caplog):
    from emridispatch.workbench_plots import plot_char_strain

    model.add_noise = False
    with caplog.at_level("WARNING"):
        plot_char_strain(model, {"e": 0.1}, show=("data",))

    assert "add_noise" in caplog.text


def test_snr_accumulation_optimal_is_monotonic(model):
    from emridispatch.workbench_plots import plot_snr_accumulation

    fig, axes = plot_snr_accumulation(model, {"e": 0.1})
    line = np.atleast_1d(axes)[0].get_lines()[0]
    assert np.all(np.diff(line.get_ydata()) >= -1e-12)


def test_snr_accumulation_detected_not_required_monotonic(model):
    from emridispatch.workbench_plots import plot_snr_accumulation

    fig, axes = plot_snr_accumulation(model, {"e": 0.1}, show=("data",))
    line = [ln for ln in np.atleast_1d(axes)[0].get_lines()
            if ln.get_label() == "data"][0]
    assert np.any(np.diff(line.get_ydata()) < 0)


def test_snr_accumulation_injection_only_builds_one_template(model):
    from emridispatch.workbench_plots import plot_snr_accumulation

    fig, axes = plot_snr_accumulation(model, show=("injection",))
    labels = [ln.get_label() for ln in np.atleast_1d(axes)[0].get_lines()]
    assert "injection" in labels
    assert model.calls == 1


def test_time_domain_residual_and_plain(model):
    from emridispatch.workbench_plots import plot_time_domain

    fig, axes = plot_time_domain(model, {"e": 0.1})
    assert len(np.atleast_1d(axes)) == len(CHANNELS)

    fig, axes = plot_time_domain(model, {"e": 0.1}, residual=True)
    assert len(np.atleast_1d(axes)) == len(CHANNELS)


def test_time_domain_shows_noise_trace(model):
    from emridispatch.workbench_plots import plot_time_domain

    fig, axes = plot_time_domain(model, {"e": 0.1}, show=("template", "noise"))
    labels = [line.get_label() for line in np.atleast_1d(axes)[0].get_lines()]
    assert "noise" in labels


def test_time_domain_whiten_changes_values(model):
    from emridispatch.workbench_plots import plot_time_domain

    fig_plain, axes_plain = plot_time_domain(model, {"e": 0.1})
    fig_white, axes_white = plot_time_domain(model, {"e": 0.1}, whiten=True)

    plain = np.atleast_1d(axes_plain)[0].get_lines()[0].get_ydata()
    white = np.atleast_1d(axes_white)[0].get_lines()[0].get_ydata()
    assert len(np.atleast_1d(axes_white)) == len(np.atleast_1d(axes_plain))
    assert not np.allclose(plain, white)


def test_time_domain_data_only_uses_reported_length(model):
    from emridispatch.workbench_plots import plot_time_domain

    fig, axes = plot_time_domain(model, show=("data",))
    line = np.atleast_1d(axes)[0].get_lines()[0]
    assert len(line.get_xdata()) == N_TIME


def test_time_domain_data_trace_reconstructs_the_injected_strain(model):
    """A shape assertion cannot see a mis-spaced transform, so check the values."""
    from emridispatch.workbench_plots import plot_time_domain

    times = np.arange(N_TIME) * model.delta_t
    known = np.zeros((NCHAN, N_TIME))
    known[:, :N_TIME_NATIVE] = np.asarray([
        np.sin(2 * np.pi * 3.0e-4 * (1.0 + c) * times[:N_TIME_NATIVE])
        for c in range(NCHAN)])
    spectrum = np.fft.rfft(known, axis=-1) * model.delta_t
    assert spectrum.shape == (NCHAN, NF)
    model.data_residual_array = _Domain(
        spectrum, np.arange(NF) * DF, DF, n_time=N_TIME)

    fig, axes = plot_time_domain(model, {"e": 0.1}, show=("template", "data"))
    line = [ln for ln in np.atleast_1d(axes)[0].get_lines()
            if ln.get_label() == "data"][0]
    recovered = line.get_ydata()

    assert len(recovered) == N_TIME_NATIVE
    assert np.allclose(recovered, known[0, :N_TIME_NATIVE], atol=1e-10)


def test_time_frequency_data_only(model):
    from emridispatch.workbench_plots import plot_time_frequency

    fig, axes = plot_time_frequency(model, show=("data",))
    assert len(np.atleast_1d(axes)) == len(CHANNELS)


def _mesh(axes):
    return np.atleast_1d(axes)[0].collections[0]


def test_default_nperseg_tracks_series_length():
    from emridispatch.workbench_plots import _default_nperseg

    assert _default_nperseg(3153814) == 1024
    assert _default_nperseg(10000) == 64
    assert _default_nperseg(4) == 4
    lengths = [100, 10000, 1000000, 3153814]
    windows = [_default_nperseg(n) for n in lengths]
    assert windows == sorted(windows)
    assert all(w <= n for w, n in zip(windows, lengths))


def test_time_frequency_drops_the_zero_frequency_bin(model):
    from emridispatch.workbench_plots import plot_time_frequency

    fig, axes = plot_time_frequency(model, {"e": 0.1})
    axis = np.atleast_1d(axes)[0]
    assert _mesh(axes).get_coordinates()[..., 1].min() > 0.0
    assert axis.get_ylim()[0] > 0.0
    assert axis.get_ylim()[1] > axis.get_ylim()[0]


def test_time_frequency_frames_the_signal_band(model):
    """Auto limits must bracket the frequency carrying power.

    The stub's 1 mHz tone is 1.2 cycles in 124 samples; 20 mHz is resolvable.
    """
    from emridispatch.workbench_plots import plot_time_frequency

    f_tone = 0.02

    def tone(params):
        times = np.arange(N_TIME_NATIVE) * model.delta_t
        return times, np.asarray(
            [np.sin(2 * np.pi * f_tone * times) for _ in range(NCHAN)])

    model.generate_time_domain = tone
    fig, axes = plot_time_frequency(model, {"e": 0.1},
                                    stft_kwargs={"nperseg": 32})
    lo, hi = np.atleast_1d(axes)[0].get_ylim()
    assert lo < f_tone < hi
    assert hi / lo < 100.0


def _plunging_tone(model, stop_fraction=0.5, f_tone=0.02):
    """Install a 20 mHz tone that stops partway through, like an early plunge."""
    times = np.arange(N_TIME_NATIVE) * model.delta_t
    strain = np.asarray(
        [np.sin(2 * np.pi * f_tone * times) for _ in range(NCHAN)])
    strain[:, int(stop_fraction * N_TIME_NATIVE):] = 0.0
    model.generate_time_domain = lambda params: (times, strain)
    return times


def test_time_frequency_trims_to_signal_support(model):
    from emridispatch.workbench_plots import plot_time_frequency

    times = _plunging_tone(model, stop_fraction=0.5)
    fig, axes = plot_time_frequency(model, {"e": 0.1},
                                    stft_kwargs={"nperseg": 16})
    lo, hi = np.atleast_1d(axes)[0].get_xlim()
    assert hi < 0.75 * times[-1]
    assert lo < 0.25 * times[-1] < hi


def test_time_frequency_span_is_the_union_over_panels(model):
    """A trace with support everywhere must widen the shared x axis back out."""
    from emridispatch.workbench_plots import plot_time_frequency

    times = _plunging_tone(model, stop_fraction=0.5)
    trimmed, axes_trimmed = plot_time_frequency(
        model, {"e": 0.1}, stft_kwargs={"nperseg": 16})
    widened, axes_widened = plot_time_frequency(
        model, {"e": 0.1}, show=("template", "data"),
        stft_kwargs={"nperseg": 16})

    assert (np.atleast_1d(axes_widened)[0].get_xlim()[1]
            > np.atleast_1d(axes_trimmed)[0].get_xlim()[1])
    for axis in np.atleast_1d(axes_widened):
        assert axis.get_xlim() == pytest.approx(
            np.atleast_1d(axes_widened)[0].get_xlim())


def test_time_frequency_tlim_overrides_auto_span(model):
    from emridispatch.workbench_plots import plot_time_frequency

    _plunging_tone(model, stop_fraction=0.5)
    fig, axes = plot_time_frequency(model, {"e": 0.1}, tlim=(100.0, 900.0),
                                    stft_kwargs={"nperseg": 16})
    assert np.atleast_1d(axes)[0].get_xlim() == pytest.approx((100.0, 900.0))


def test_time_frequency_linear_scale_keeps_full_span(model):
    from emridispatch.workbench_plots import plot_time_frequency

    times = _plunging_tone(model, stop_fraction=0.5)
    fig, axes = plot_time_frequency(model, {"e": 0.1}, dynamic_range=None,
                                    stft_kwargs={"nperseg": 16})
    assert np.atleast_1d(axes)[0].get_xlim()[1] >= times[-1]


def test_time_frequency_flim_overrides_auto_limits(model):
    from emridispatch.workbench_plots import plot_time_frequency

    fig, axes = plot_time_frequency(model, {"e": 0.1}, flim=(1e-3, 2e-2))
    assert np.atleast_1d(axes)[0].get_ylim() == pytest.approx((1e-3, 2e-2))


def test_time_frequency_linear_scale_on_request(model):
    from emridispatch.workbench_plots import plot_time_frequency

    fig, axes = plot_time_frequency(model, {"e": 0.1}, dynamic_range=None)
    norm = _mesh(axes).norm
    assert isinstance(norm, Normalize)
    assert not isinstance(norm, LogNorm)


def test_time_frequency_caller_norm_wins(model):
    from emridispatch.workbench_plots import plot_time_frequency

    supplied = Normalize(vmin=0.0, vmax=1.0)
    fig, axes = plot_time_frequency(model, {"e": 0.1}, norm=supplied)
    assert _mesh(axes).norm is supplied


def test_time_frequency_zero_amplitude_trace_falls_back_to_linear(model):
    from emridispatch.workbench_plots import plot_time_frequency

    model.generate_time_domain = lambda params: (
        np.arange(N_TIME_NATIVE) * model.delta_t,
        np.zeros((NCHAN, N_TIME_NATIVE)))
    fig, axes = plot_time_frequency(model, {"e": 0.1})
    assert not isinstance(_mesh(axes).norm, LogNorm)


def test_time_frequency_colorbar_can_be_suppressed(model):
    from emridispatch.workbench_plots import plot_time_frequency

    with_bar, _ = plot_time_frequency(model, {"e": 0.1}, colorbar=True)
    without, _ = plot_time_frequency(model, {"e": 0.1}, colorbar=False)
    assert len(with_bar.axes) == len(without.axes) + len(CHANNELS)


def test_plots_accept_caller_axes(model):
    from emridispatch.workbench_plots import plot_char_strain

    fig, ax = plt.subplots(len(CHANNELS), 1)
    out_fig, out_axes = plot_char_strain(model, ax=ax)
    assert out_fig is fig


def test_show_notch_skipped_when_mask_absent(model):
    from emridispatch.workbench_plots import plot_char_strain

    model._psd_notch_mask = None
    fig, axes = plot_char_strain(model, show_notch=True)
    assert fig is not None
