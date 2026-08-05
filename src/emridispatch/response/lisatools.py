"""EMRI injection + matched-filter likelihood via lisa-analysis-tools + FEW.

Optional extra `emridispatch[lisatools]`; imports are deferred to
construction, so this module always imports cleanly without it.
"""

import logging
from types import SimpleNamespace

import numpy as np
from scipy.fft import next_fast_len

from emridispatch.noise import (
    MIN_FREQ, load_sensitivity_table, noise_sens_kwargs, sensitivity_spec)
from emridispatch.parameters import few_params, physical_from_vector
from emridispatch.response import InjectionModel

logger = logging.getLogger(__name__)


class PSDNullError(RuntimeError):
    """The notched SNR depends on the notch width, so no notch recovers a
    trustworthy value."""


def _import_lisatools():
    try:
        from lisatools.datacontainer import DataResidualArray
        from lisatools.domains import FDSettings, TDSettings
        from lisatools.sensitivity import SensitivityMatrix
        from lisatools.analysiscontainer import AnalysisContainer
        from lisatools.diagnostic import inner_product, noise_likelihood_term
        from lisatools.response import ResponseWrapper
        from lisatools.detector import EqualArmlengthOrbits

        sens_table = load_sensitivity_table()
    except ImportError as err:
        raise ImportError(
            "lisa-analysis-tools is required for the 'lisatools' response "
            "model. Install with `pip install emridispatch[lisatools]`."
        ) from err
    return SimpleNamespace(
        DataResidualArray=DataResidualArray,
        TDSettings=TDSettings,
        FDSettings=FDSettings,
        SensitivityMatrix=SensitivityMatrix,
        AnalysisContainer=AnalysisContainer,
        inner_product=inner_product,
        noise_likelihood_term=noise_likelihood_term,
        ResponseWrapper=ResponseWrapper,
        EqualArmlengthOrbits=EqualArmlengthOrbits,
        sens_table=sens_table,
    )


def _import_few():
    try:
        from few.waveform.waveform import GenerateEMRIWaveform
    except ImportError as err:
        raise ImportError(
            "FastEMRIWaveforms is required for the 'lisatools' response "
            "model. Install with `pip install emridispatch[lisatools]`."
        ) from err
    return SimpleNamespace(GenerateEMRIWaveform=GenerateEMRIWaveform)


class _DirectEMRIWaveform:
    def __init__(self, gen, T, dt):
        self.gen = gen
        self.T = T
        self.dt = dt

    def __call__(self, *params):
        h = self.gen(*params, T=self.T, dt=self.dt)
        return [h.real, -h.imag]


def _build_waveform_and_sens(lt, few, tdi, channel_list, duration, delta_t):
    if tdi == "off":
        if channel_list is not None:
            logger.info("tdi off: ignoring data.channels %s", channel_list)
        gen = few.GenerateEMRIWaveform(
            "FastKerrEccentricEquatorialFlux",
            sum_kwargs=dict(pad_output=True), return_list=False,
            frame="detector")
        sens, channels = sensitivity_spec("off", None, lt.sens_table)
        return _DirectEMRIWaveform(gen, duration, delta_t), channels, sens

    sens, channels = sensitivity_spec(tdi, channel_list, lt.sens_table)
    gen = few.GenerateEMRIWaveform(
        "FastKerrEccentricEquatorialFlux", sum_kwargs=dict(pad_output=True))
    wf = lt.ResponseWrapper(
        gen, duration, delta_t, index_lambda=8, index_beta=7,
        t0=30000.0, flip_hx=True, remove_sky_coords=False,
        is_ecliptic_latitude=False, order=25, tdi=tdi,
        tdi_chan="".join(channels), orbits=lt.EqualArmlengthOrbits())
    return wf, channels, sens


class EMRIInjectionGenerator:
    def __init__(
        self,
        injection_parameters,
        duration=1,
        delta_t=10,
        injection_snr=None,
        channel_list=None,
        tdi="2nd generation",
        foreground=True,
        add_noise=False,
        noise_seed=0,
        pad_fft=True,
        psd_notch=1e-5,
        psd_notch_depth=2.0,
        psd_notch_strict=True,
    ):
        self._lt = _import_lisatools()
        self._few = _import_few()

        self.injection_parameters = dict(injection_parameters)
        self.injection_snr = injection_snr
        self.delta_t = delta_t
        self.duration = duration
        self.add_noise = add_noise
        self.noise_seed = noise_seed
        self.tdi = tdi
        self.foreground = foreground
        self.pad_fft = pad_fft
        self._fft_length = None
        self.psd_notch = float(psd_notch)
        self.psd_notch_depth = float(psd_notch_depth)
        self.psd_notch_strict = bool(psd_notch_strict)
        self._psd_notch_mask = None
        self._notch_drift = None

        self.waveform_generator, self.channel_list, self.sensitivity_list = (
            _build_waveform_and_sens(
                self._lt, self._few, self.tdi, channel_list, self.duration,
                self.delta_t))

        if self.tdi == "off":
            logger.info(
                "tdi: off (direct channels %s, sky-averaged LISASens)",
                ", ".join(self.channel_list))
        else:
            _response = self.waveform_generator
            logger.info(
                "backends: response=%s tdi=%s few=%s",
                _response.backend.name,
                _response.response_model.backend.name,
                _response.waveform_gen.waveform_generator.backend.name,
            )

        self._build_injection()

        self._noise_term = self._lt.noise_likelihood_term(self.sensitivity_matrix)
        self._d_d = self._lt.inner_product(
            self.data_residual_array,
            self.data_residual_array,
            psd=self.sensitivity_matrix,
            normalize=False,
        )
        self._lnlike_const = self._noise_term - 0.5 * self._d_d


    def _pad_to_fft_length(self, channel_strain, xp):
        n = channel_strain.shape[-1]
        if self._fft_length is None:
            self._fft_length = next_fast_len(n, True) if self.pad_fft else n
            logger.info("fft length: %d -> %d samples (+%.2f%%)", n,
                        self._fft_length, 100.0 * (self._fft_length - n) / n)
        n_pad = self._fft_length
        if n == n_pad:
            return channel_strain
        if n > n_pad:
            logger.error("template length %d exceeds the fixed FFT length %d",
                         n, n_pad)
            raise ValueError(
                f"template length {n} exceeds the fixed FFT length {n_pad}")
        padded = xp.zeros(channel_strain.shape[:-1] + (n_pad,),
                          dtype=channel_strain.dtype)
        padded[..., :n] = channel_strain
        return padded

    def _psd_null_mask(self, f_arr, sens_mat, block=4096, _width=None):
        width = self.psd_notch if _width is None else _width
        if width <= 0.0:
            return None
        f = np.asarray(f_arr.get() if hasattr(f_arr, "get") else f_arr)
        S = np.asarray(sens_mat.get() if hasattr(sens_mat, "get") else sens_mat)
        S = S.reshape(-1, S.shape[-1])
        nf = S.shape[-1]
        nblk = int(np.ceil(nf / block))
        pad = nblk * block - nf
        bad = np.zeros(nf, dtype=bool)
        with np.errstate(all="ignore"):
            for row in S:
                v = np.log10(np.where(np.isfinite(row) & (row > 0), row, np.nan))
                vp = np.concatenate([v, np.full(pad, np.nan)]).reshape(nblk, block)
                med = np.nanmedian(vp, axis=1)[:, None]
                bad |= (vp < med - self.psd_notch_depth).reshape(-1)[:nf]
        if not bad.any():
            return None
        idx = np.flatnonzero(bad)
        groups = [g for g in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
                  if g[0] != 0]
        if not groups:
            return None
        mask = np.zeros(nf, dtype=bool)
        for g in groups:
            a = np.searchsorted(f, f[g[0]] - width, side="left")
            b = np.searchsorted(f, f[g[-1]] + width, side="right")
            mask[a:b] = True
        if _width is None:
            shown = ", ".join(f"{f[g].mean():.6g}" for g in groups[:4])
            if len(groups) > 4:
                shown += f", ... (+{len(groups) - 4} more)"
            logger.info(
                "psd notch: %d null(s) at %s Hz -> %d/%d bins masked "
                "(half-width %g Hz, depth %g decades)",
                len(groups), shown, int(mask.sum()), nf, width,
                self.psd_notch_depth)
        return mask

    def _apply_psd_notch(self, data):
        sens = self.sensitivity_matrix
        mask = self._psd_null_mask(data.settings.f_arr, sens.sens_mat)
        if mask is None:
            return
        invC = sens.invC
        try:
            import cupy as _cp

            xp = _cp.get_array_module(invC)
        except ImportError:
            xp = np
        invC[..., xp.asarray(mask)] = 0.0
        sens.invC = invC
        self._psd_notch_mask = mask
        self._check_notch_stability(data, mask)

    def _check_notch_stability(self, data, mask, factor=10.0, tol=0.01):
        wide = self._psd_null_mask(
            data.settings.f_arr, self.sensitivity_matrix.sens_mat,
            _width=self.psd_notch * factor)
        if wide is None:
            return
        d = data.data_res_arr.arr
        try:
            import cupy as _cp

            xp = _cp.get_array_module(d)
        except ImportError:
            xp = np
        S = self.sensitivity_matrix.sens_mat
        good = xp.isfinite(S) & (S > 0)
        power = xp.where(good, xp.abs(d) ** 2 / xp.where(good, S, 1.0), 0.0)
        keep_n = ~xp.asarray(mask)
        keep_w = ~xp.asarray(wide)
        s_n = float(xp.sum(power * keep_n))
        s_w = float(xp.sum(power * keep_w))
        if not np.isfinite(s_n) or s_n <= 0.0:
            self._notch_drift = float("nan")
            logger.warning(
                "psd notch check skipped: injection has no SNR outside the "
                "notched bands (<d|d> = %r). The notch stability of this "
                "injection is unverified.", s_n)
            return
        drift = abs(s_w / s_n - 1.0)
        self._notch_drift = drift
        if drift <= tol:
            logger.info("psd notch stable: %gx widening shifts SNR^2 by %.3f%%",
                        factor, 100.0 * drift)
            return
        msg = (
            f"psd notch unstable: {factor:g}x widening shifts SNR^2 by "
            f"{100.0 * drift:.1f}% (tol {100.0 * tol:.1f}%). Try `data.tdi: 1st` "
            f"or `data.tdi: off`; `data.psd_notch_strict: false` proceeds.")
        if self.psd_notch_strict:
            raise PSDNullError(msg)
        logger.warning("psd notch UNSTABLE (psd_notch_strict=false): %s", msg)

    def _fd_settings(self, td_settings):
        return self._lt.FDSettings(
            td_settings.N // 2 + 1, 1.0 / (td_settings.N * td_settings.dt),
            min_freq=MIN_FREQ, force_backend=td_settings.force_backend)

    def _produce_data_residual_array(self, params=None):
        if params is None:
            params = self.injection_parameters

        _channel_strain = self.waveform_generator(*few_params(params).values())

        try:
            import cupy as _cp

            xp = _cp.get_array_module(_channel_strain[0])
        except ImportError:
            xp = np
        channel_strain = xp.asarray([xp.asarray(AET) for AET in _channel_strain])
        channel_strain = self._pad_to_fft_length(channel_strain, xp)
        td_settings = self._lt.TDSettings(channel_strain.shape[-1], self.delta_t)
        return self._lt.DataResidualArray(
            channel_strain, input_signal_domain=td_settings,
            signal_domain=self._fd_settings(td_settings))

    def _build_injection(self):
        uncalibrated = self._produce_data_residual_array()
        self.sensitivity_matrix = self._lt.SensitivityMatrix(
            uncalibrated.settings,
            self.sensitivity_list,
            **noise_sens_kwargs(self.duration, self.foreground),
        )
        self._apply_psd_notch(uncalibrated)
        container = self._lt.AnalysisContainer(
            uncalibrated,
            self.sensitivity_matrix,
            signal_gen=self.waveform_generator,
        )

        # rescale distance to get injection SNR
        if self.injection_snr is not None:
            self.injection_parameters["luminosity_distance"] /= (
                self.injection_snr / container.snr()
            )

            self.data_residual_array = self._produce_data_residual_array()

            assert np.all(
                self.data_residual_array.settings.f_arr
                == uncalibrated.settings.f_arr
            )

            self.analysis_container = self._lt.AnalysisContainer(
                self.data_residual_array,
                self.sensitivity_matrix,
                signal_gen=self.waveform_generator,
            )
            assert np.isclose(self.injection_snr, self.analysis_container.snr())
        else:
            self.data_residual_array = uncalibrated
            self.analysis_container = container

        self.optimal_snr = float(np.real(self.analysis_container.snr()))

        if self.add_noise:
            self._add_noise_realization()

    def _add_noise_realization(self):
        """Add a Gaussian noise realization drawn from the sensitivity PSD.

        Per good bin: n_tilde = sqrt(S(f)*T/4)*(x+iy), x,y~N(0,1), giving
        E[<n|n>]=2 per bin per channel. Non-finite/<=0 PSD bins get none.
        """
        fd = self.data_residual_array.data_res_arr  # FDSignal
        xp = fd.xp
        _sens_mat = self.sensitivity_matrix.sens_mat
        S = (_sens_mat.get() if hasattr(_sens_mat, "get")
             else np.asarray(_sens_mat))                   # (nchan, nf), host
        nchan, nf = S.shape
        T = 1.0 / float(fd.df)
        good = np.isfinite(S).all(axis=0) & (S > 0).all(axis=0)

        rng = np.random.default_rng(self.noise_seed)
        noise = np.zeros((nchan, nf), dtype=complex)
        ngood = int(good.sum())
        noise[:, good] = np.sqrt(S[:, good] * T / 4.0) * (
            rng.standard_normal((nchan, ngood))
            + 1j * rng.standard_normal((nchan, ngood))
        )

        fd.arr[:] = fd.arr + xp.asarray(noise)

        # Rebuild the container so anything derived from the data sees the noise.
        self.analysis_container = self._lt.AnalysisContainer(
            self.data_residual_array,
            self.sensitivity_matrix,
            signal_gen=self.waveform_generator,
        )
        logger.info("noise: added PSD realization (seed=%s, %d/%d bins)",
                    self.noise_seed, ngood, nf)

    def generate_signal(self, params):
        return self._produce_data_residual_array(params=params)

    def generate_time_domain(self, params):
        """(times, strain) for a physical-parameter dict, on the host.

        strain has shape (nchannels, N) at native length, before FFT
        padding.
        """
        channels = self.waveform_generator(*few_params(params).values())
        strain = np.asarray([
            np.asarray(c.get() if hasattr(c, "get") else c) for c in channels])
        times = np.arange(strain.shape[-1]) * self.delta_t
        return times, strain

    def evaluate_likelihood(self, template_params, full=False):
        """Log-likelihood for a template.

        Default returns only <d|h> - 0.5<h|h>. full=True returns the
        absolute ln L = noise + (-0.5)(<d|d> + <h|h> - 2<d|h>).
        """
        if isinstance(template_params, dict):
            sig_dat_array = self.generate_signal(template_params)
        elif isinstance(template_params, self._lt.DataResidualArray):
            sig_dat_array = template_params
        else:
            raise TypeError(
                f"template_params must be a dict or DataResidualArray, got "
                f"{type(template_params).__name__}")

        d_h = self._lt.inner_product(
            self.data_residual_array, sig_dat_array,
            psd=self.sensitivity_matrix, normalize=False,
        )
        h_h = self._lt.inner_product(
            sig_dat_array, sig_dat_array,
            psd=self.sensitivity_matrix, normalize=False,
        )
        varying_term = d_h - 0.5 * h_h
        if full:
            # Full ln L = noise + (-1/2)(<d|d> + <h|h> - 2<d|h>)
            #           = (_noise_term - 1/2 <d|d>) + (<d|h> - 1/2 <h|h>).
            return self._lnlike_const + varying_term
        return varying_term


class LisatoolsEMRILikelihood(EMRIInjectionGenerator, InjectionModel):

    log_lnlike_failures = False

    def __init__(self, injection_parameters, **kwargs):
        super().__init__(injection_parameters, **kwargs)
        self.lnlike_failures = {}

    @classmethod
    def from_config(cls, cfg):
        _channels = cfg.data.channels
        return cls(
            dict(cfg.injection),
            duration=cfg.data.duration, delta_t=cfg.data.delta_t,
            injection_snr=cfg.data.inj_snr,
            channel_list=None if _channels is None else list(_channels),
            tdi=str(cfg.data.tdi),
            foreground=bool(cfg.data.foreground),
            add_noise=bool(cfg.data.add_noise),
            noise_seed=int(cfg.data.noise_seed),
            pad_fft=bool(cfg.data.pad_fft),
            psd_notch=float(cfg.data.psd_notch),
            psd_notch_depth=float(cfg.data.psd_notch_depth),
            psd_notch_strict=bool(cfg.data.psd_notch_strict),
        )

    def __call__(self, params):
        """ln L for a 12-D sampling vector.

        The template is built over injection_parameters, not a fixed default.
        The vector overwrites every sampled row; the unsampled ones carry no
        independent content for the current equatorial models, but a model that
        does use them must see the injection's values, not hard-coded ones.
        """
        try:
            template_params = physical_from_vector(params, self.injection_parameters)
        except Exception as err:
            self._note_lnlike_failure("parameters", err)
            return -np.inf

        try:
            result = self.evaluate_likelihood(template_params)

            if hasattr(result, "get"):
                result = result.get()
            result = np.real(result)
        except Exception as err:
            self._note_lnlike_failure("waveform", err)
            return -np.inf

        if np.isfinite(result):
            return float(result)
        else:
            return -np.inf

    def _note_lnlike_failure(self, stage, err):
        """Count a swallowed __call__ failure, keyed "<stage>: <ExceptionType>".

        Silent unless log_lnlike_failures is set, which then logs the first of
        each kind.
        """
        failures = getattr(self, "lnlike_failures", None)
        if failures is None:
            failures = self.lnlike_failures = {}
        key = f"{stage}: {type(err).__name__}"
        failures[key] = failures.get(key, 0) + 1
        if self.log_lnlike_failures and failures[key] == 1:
            logger.warning(
                "lnlike %s failure, returning -inf: %s: %s. Later failures of "
                "this kind are counted in lnlike_failures, not logged",
                stage, type(err).__name__, err, exc_info=True)
