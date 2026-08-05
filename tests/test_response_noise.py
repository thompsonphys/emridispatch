import numpy as np
import pytest

pytest.importorskip("lisatools.datacontainer")

from lisatools.datacontainer import DataResidualArray
from lisatools.domains import TDSettings
from lisatools.sensitivity import LISASens, SensitivityMatrix

from emridispatch.response.lisatools import EMRIInjectionGenerator

BACKENDS = [
    "cpu",
    pytest.param(
        None,
        marks=pytest.mark.skipif(
            not TDSettings(8, 1.0).backend.uses_cupy,
            reason="default lisatools backend is not GPU/cupy",
        ),
    ),
]


class _FakeAnalysisContainer:
    def __init__(self, data, sens, signal_gen=None):
        self.data = data
        self.sens = sens


def _make_generator(force_backend, seed=3):
    n, dt = 512, 10.0
    settings = TDSettings(n, dt, force_backend=force_backend)
    strain = settings.xp.asarray(
        1e-21 * np.random.default_rng(0).standard_normal((2, n)))
    dra = DataResidualArray(strain, input_signal_domain=settings)
    sens = SensitivityMatrix(dra.settings, [LISASens, LISASens], model="scirdv1")

    gen = object.__new__(EMRIInjectionGenerator)
    gen.data_residual_array = dra
    gen.sensitivity_matrix = sens
    gen.noise_seed = seed
    gen.waveform_generator = None

    class _LT:
        AnalysisContainer = _FakeAnalysisContainer

    gen._lt = _LT()
    return gen


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _data(gen):
    return _host(gen.data_residual_array.data_res_arr.arr)


@pytest.mark.parametrize("force_backend", BACKENDS)
def test_add_noise_realization_changes_data(force_backend):
    gen = _make_generator(force_backend)
    before = _data(gen).copy()

    gen._add_noise_realization()

    after = _data(gen)
    assert not np.array_equal(before, after)
    assert np.isfinite(after).all()


@pytest.mark.parametrize("force_backend", BACKENDS)
def test_add_noise_realization_is_seeded(force_backend):
    a = _make_generator(force_backend, seed=11)
    b = _make_generator(force_backend, seed=11)
    c = _make_generator(force_backend, seed=12)
    for g in (a, b, c):
        g._add_noise_realization()

    np.testing.assert_array_equal(_data(a), _data(b))
    assert not np.array_equal(_data(a), _data(c))


@pytest.mark.parametrize("force_backend", BACKENDS)
def test_add_noise_realization_leaves_bad_psd_bins_untouched(force_backend):
    gen = _make_generator(force_backend)
    S = _host(gen.sensitivity_matrix.sens_mat)
    bad = ~(np.isfinite(S).all(axis=0) & (S > 0).all(axis=0))
    assert bad.any()

    before = _data(gen).copy()
    gen._add_noise_realization()

    np.testing.assert_array_equal(before[:, bad], _data(gen)[:, bad])


@pytest.mark.parametrize("force_backend", BACKENDS)
def test_add_noise_realization_amplitude_matches_psd(force_backend):
    gen = _make_generator(force_backend)
    before = _data(gen).copy()
    gen._add_noise_realization()

    S = _host(gen.sensitivity_matrix.sens_mat)
    good = np.isfinite(S).all(axis=0) & (S > 0).all(axis=0)
    T = 1.0 / float(gen.data_residual_array.data_res_arr.df)

    drawn = (_data(gen) - before)[:, good]
    ratio = np.abs(drawn) / np.sqrt(S[:, good] * T / 4.0)
    assert 0.5 < ratio.mean() < 2.0


@pytest.mark.parametrize("force_backend", BACKENDS)
def test_add_noise_realization_keeps_data_on_device(force_backend):
    gen = _make_generator(force_backend)
    expected = type(gen.data_residual_array.data_res_arr.arr)

    gen._add_noise_realization()

    assert type(gen.data_residual_array.data_res_arr.arr) is expected


def test_add_noise_realization_matches_across_backends():
    cpu = _make_generator("cpu", seed=7)
    if not TDSettings(8, 1.0).backend.uses_cupy:
        pytest.skip("default lisatools backend is not GPU/cupy")
    gpu = _make_generator(None, seed=7)

    cpu._add_noise_realization()
    gpu._add_noise_realization()

    np.testing.assert_allclose(_data(cpu), _data(gpu), rtol=1e-6)
