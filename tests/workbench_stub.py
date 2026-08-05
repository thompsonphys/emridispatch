"""Duck-typed stand-in for a lisatools injection model.

Diagonal sensitivity matrix, deterministic arrays; no lisatools, FEW,
or GPU needed.
"""

from types import SimpleNamespace

import numpy as np

NCHAN = 2
NF = 64
DF = 1.0e-4
CHANNELS = ["A", "E"]

N_TIME = 127
N_TIME_NATIVE = 124


def _arr(obj):
    """Channel array for a container, an inner domain, or a bare array."""
    inner = getattr(obj, "data_res_arr", obj)
    return np.asarray(getattr(inner, "arr", inner))


def _inner(a, b, inv_psd, differential):
    arr_a, arr_b = _arr(a), _arr(b)
    total = 0.0
    for i in range(arr_a.shape[0]):
        start = 1 if np.isnan(inv_psd[i][0]) else 0
        y = (np.real(arr_a[i][start:].conj() * arr_b[i][start:])
             * inv_psd[i][start:])
        total += 4.0 * float(np.sum(y)) * differential
    return total


_NESTED = ("arr", "f_arr", "df")


class _InnerDomain:
    """The inner lisatools domain object: the only level that owns the arrays."""

    def __init__(self, arr, f_arr, df):
        self.arr = np.asarray(arr)
        self.f_arr = np.asarray(f_arr)
        self.df = df

    def __getitem__(self, index):
        return self.arr[index]

    @property
    def settings(self):
        return SimpleNamespace(N=self.arr.shape[-1])


class _Domain:
    """The outer DataResidualArray-shaped wrapper around an _InnerDomain.

    arr, f_arr and df raise here as they do on the real wrapper, reachable only
    through .data_res_arr; .settings and .init_kwargs live at this level.
    """

    def __init__(self, arr, f_arr, df, n_time=None):
        self.data_res_arr = _InnerDomain(arr, f_arr, df)
        self.init_kwargs = {
            "input_signal_domain": SimpleNamespace(N=n_time)
            if n_time is not None else None}

    def __getattr__(self, name):
        if name in _NESTED:
            raise AttributeError(
                f"DataResidualArray has no usable {name}; read it from "
                f".data_res_arr")
        raise AttributeError(name)

    def __getitem__(self, index):
        return self.data_res_arr.arr[index]

    @property
    def settings(self):
        return self.data_res_arr.settings


class _Sens:
    def __init__(self, sens_mat, differential):
        self.sens_mat = np.asarray(sens_mat)
        self.invC = 1.0 / self.sens_mat
        self.differential_component = differential

    def __getitem__(self, index):
        return self.sens_mat[index]


class _Container:
    def __init__(self, data, sens):
        self._data = data
        self._sens = sens

    def template_snr(self, template, phase_maximize=False, **kwargs):
        h_h = _inner(template, template, self._sens.invC,
                     self._sens.differential_component)
        d_h = _inner(self._data, template, self._sens.invC,
                     self._sens.differential_component)
        opt = np.sqrt(h_h)
        return (opt, abs(d_h) / opt if phase_maximize else d_h / opt)


class StubModel:
    """Minimal InjectionModel-shaped object with a diagonal PSD."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        f_arr = np.arange(NF) * DF
        self.channel_list = list(CHANNELS)
        self.delta_t = 10.0
        self.duration = 0.3
        self.injection_parameters = {"x": 1.0, "phi_theta": 0.5}
        sens = np.abs(rng.standard_normal((NCHAN, NF))) + 1.0
        self.sensitivity_matrix = _Sens(sens, DF)
        data = rng.standard_normal((NCHAN, NF)) + 1j * rng.standard_normal((NCHAN, NF))
        self.data_residual_array = _Domain(data, f_arr, DF, n_time=N_TIME)
        self.analysis_container = _Container(self.data_residual_array,
                                             self.sensitivity_matrix)
        self._templates = {}
        self._rng = rng
        self.calls = 0
        self._psd_notch_mask = None
        self.add_noise = True
        self.noise_seed = 0

    def generate_signal(self, params):
        self.calls += 1
        key = tuple(sorted((k, float(v)) for k, v in params.items()))
        if key not in self._templates:
            rng = np.random.default_rng(abs(hash(key)) % (2**32))
            arr = (rng.standard_normal((NCHAN, NF))
                   + 1j * rng.standard_normal((NCHAN, NF)))
            self._templates[key] = _Domain(
                arr, self.data_residual_array.data_res_arr.f_arr, DF)
        return self._templates[key]

    def generate_time_domain(self, params):
        """(times, strain) at the generator's native length, N_TIME_NATIVE.

        Shorter than the padded N_TIME the frequency-domain data was built at,
        as in the real stack, so a length mismatch is observable.
        """
        n = N_TIME_NATIVE
        times = np.arange(n) * self.delta_t
        f0 = 1.0e-3 * (1.0 + params.get("e", 0.0))
        strain = np.asarray([
            np.sin(2 * np.pi * f0 * (1 + 0.2 * c) * times) for c in range(NCHAN)])
        return times, strain

    def evaluate_likelihood(self, payload, full=False):
        template = (self.generate_signal(payload) if isinstance(payload, dict)
                    else payload)
        inv = self.sensitivity_matrix.invC
        d_h = _inner(self.data_residual_array, template, inv, DF)
        h_h = _inner(template, template, inv, DF)
        varying = d_h - 0.5 * h_h
        return varying + 1000.0 if full else varying
