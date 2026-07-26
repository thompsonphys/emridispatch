"""Smoke tests for the workbench plots: shapes and plumbing, not appearance."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from workbench_stub import (
    CHANNELS, DF, N_TIME, N_TIME_NATIVE, NCHAN, NF, StubModel, _Domain)


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
    """The "data" trace must be the actual signal, not a rescaled length.

    The frequency-domain data lives on the padded grid (N_TIME) while the
    time-domain generator returns N_TIME_NATIVE samples. Inverse-transforming at
    the native length yields something uncorrelated with the strain, which a
    shape assertion cannot see -- so this reconstructs a known signal.
    """
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


def test_plots_accept_caller_axes(model):
    import matplotlib.pyplot as plt

    from emridispatch.workbench_plots import plot_char_strain

    fig, ax = plt.subplots(len(CHANNELS), 1)
    out_fig, out_axes = plot_char_strain(model, ax=ax)
    assert out_fig is fig


def test_show_notch_skipped_when_mask_absent(model):
    from emridispatch.workbench_plots import plot_char_strain

    model._psd_notch_mask = None
    fig, axes = plot_char_strain(model, show_notch=True)
    assert fig is not None
