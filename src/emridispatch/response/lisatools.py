"""EMRI injection + matched-filter likelihood via lisa-analysis-tools + FEW.

Optional dependency: `pip install emridispatch[lisatools]`. All lisatools imports
happen at construction time, so this module imports clean without it.
"""

import logging
from types import SimpleNamespace

import numpy as np
from scipy.fft import next_fast_len

from emridispatch.noise import (
    load_sensitivity_table, noise_sens_kwargs, sensitivity_spec)
from emridispatch.parameters import mass1_mass2_from_log_masses
from emridispatch.response import InjectionModel

logger = logging.getLogger(__name__)


class PSDNullError(RuntimeError):
    """Injection radiates at a TDI transfer-function null, where the analytic
    sensitivity is not trustworthy and no notch width recovers the true SNR."""


def _import_lisatools():
    try:
        from lisatools.datacontainer import DataResidualArray
        from lisatools.domains import FDSettings, TDSettings
        from lisatools.sensitivity import SensitivityMatrix
        from lisatools.analysiscontainer import AnalysisContainer
        from lisatools.diagnostic import inner_product, noise_likelihood_term
        from lisatools.sources.emri import EMRITDIWaveform
        from few.waveform.waveform import GenerateEMRIWaveform

        sens_table = load_sensitivity_table()
    except ImportError as err:
        raise ImportError(
            "lisa-analysis-tools (and FastEMRIWaveforms) are required for the "
            "'lisatools' response model. Install with "
            "`pip install emridispatch[lisatools]`."
        ) from err
    return SimpleNamespace(
        DataResidualArray=DataResidualArray,
        TDSettings=TDSettings,
        FDSettings=FDSettings,
        SensitivityMatrix=SensitivityMatrix,
        GenerateEMRIWaveform=GenerateEMRIWaveform,
        AnalysisContainer=AnalysisContainer,
        inner_product=inner_product,
        noise_likelihood_term=noise_likelihood_term,
        EMRITDIWaveform=EMRITDIWaveform,
        sens_table=sens_table,
    )


class _DirectEMRIWaveform:
    def __init__(self, gen, T, dt):
        self.gen = gen
        self.T = T
        self.dt = dt

    def __call__(self, *params):
        h = self.gen(*params, T=self.T, dt=self.dt)
        return [h.real, -h.imag]


def _build_waveform_and_sens(lt, tdi, channel_list, duration, delta_t):
    if tdi == "off":
        if channel_list is not None:
            logger.info("tdi off: ignoring data.channels %s", channel_list)
        gen = lt.GenerateEMRIWaveform(
            "FastKerrEccentricEquatorialFlux",
            sum_kwargs=dict(pad_output=True), return_list=False,
            frame="detector")
        sens, channels = sensitivity_spec("off", None, lt.sens_table)
        return _DirectEMRIWaveform(gen, duration, delta_t), channels, sens

    sens, channels = sensitivity_spec(tdi, channel_list, lt.sens_table)
    wf = lt.EMRITDIWaveform(
        response_kwargs=dict(
            t0=30000.0, tdi=tdi, tdi_chan="".join(channels)),
        T=duration, dt=delta_t)
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
        full_likelihood=False,
        add_noise=False,
        noise_seed=0,
        pad_fft=True,
        psd_notch=1e-5,
        psd_notch_depth=2.0,
        psd_notch_strict=True,
    ):
        self._lt = _import_lisatools()

        self.injection_parameters = injection_parameters
        self.injection_snr = injection_snr
        self.delta_t = delta_t
        self.duration = duration
        self.add_noise = add_noise
        self.noise_seed = noise_seed
        self.full_likelihood = full_likelihood
        self.tdi = tdi
        self.foreground = foreground
        self.pad_fft = pad_fft
        self._fft_length = None
        self.psd_notch = float(psd_notch)
        self.psd_notch_depth = float(psd_notch_depth)
        self.psd_notch_strict = bool(psd_notch_strict)
        self._psd_notch_mask = None

        self.waveform_generator, self.channel_list, self.sensetivity_list = (
            _build_waveform_and_sens(
                self._lt, self.tdi, channel_list, self.duration, self.delta_t))
        self.channel_string = "".join(self.channel_list)

        if self.tdi == "off":
            logger.info(
                "tdi: off (direct channels %s, sky-averaged LISASens)",
                ", ".join(self.channel_list))
        else:
            _response = self.waveform_generator.response
            logger.info(
                "backends: response=%s tdi=%s few=%s",
                _response.backend.name,
                _response.response_model.backend.name,
                _response.waveform_gen.waveform_generator.backend.name,
            )

        # Build the injection and calibrate the distance to hit injection_snr.
        self.emri_injection_generator()

        self._noise_term = self._lt.noise_likelihood_term(self.sensetivity_matrix)
        self._d_d = self._lt.inner_product(
            self.data_residual_array,
            self.data_residual_array,
            psd=self.sensetivity_matrix,
            normalize=False,
        )
        self._lnlike_const = self._noise_term - 0.5 * self._d_d

    @staticmethod
    def _get_params(params):
        M = params["mass_1"]
        a = params["a"]
        mu = params["mass_2"]
        p0 = params["p"]
        e0 = params["e"]
        x0 = params["x"]
        qK = params["q_k"]
        phiK = params["phi_k"]
        qS = params["q_s"]
        phiS = params["phi_s"]
        dist = params["luminosity_distance"]
        Phi_phi0 = params["phi_phi"]
        Phi_theta0 = params["phi_theta"]
        Phi_r0 = params["phi_r"]

        return [
            float(M),
            float(mu),
            float(a),
            float(p0),
            float(e0),
            float(x0),
            float(dist),
            float(qS),
            float(phiS),
            float(qK),
            float(phiK),
            float(Phi_phi0),
            float(Phi_theta0),
            float(Phi_r0),
        ]

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
        centres = f[bad]
        lo = np.searchsorted(f, centres - width, side="left")
        hi = np.searchsorted(f, centres + width, side="right")
        mask = np.zeros(nf, dtype=bool)
        for a, b in zip(lo, hi):
            mask[a:b] = True
        logger.info(
            "psd notch: %d null bin(s) at %s Hz -> masking %d/%d bins "
            "(half-width %g Hz, depth %g decades)",
            int(bad.sum()),
            np.array2string(np.unique(centres.round(9)), precision=7,
                            max_line_width=120),
            int(mask.sum()), nf, width, self.psd_notch_depth)
        return mask

    def _apply_psd_notch(self):
        sens = self.sensetivity_matrix
        mask = self._psd_null_mask(
            self._data_residual_array.settings.f_arr, sens.sens_mat)
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
        self._check_notch_stability(mask)

    def _check_notch_stability(self, mask, factor=10.0, tol=0.01):
        wide = self._psd_null_mask(
            self._data_residual_array.settings.f_arr,
            self.sensetivity_matrix.sens_mat,
            _width=self.psd_notch * factor)
        if wide is None:
            return
        d = self._data_residual_array.data_res_arr.arr
        try:
            import cupy as _cp

            xp = _cp.get_array_module(d)
        except ImportError:
            xp = np
        S = self.sensetivity_matrix.sens_mat
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
            f"injection radiates at the {self.tdi} TDI nulls: {factor:g}x notch "
            f"widening shifts SNR^2 by {100.0 * drift:.1f}% (tol "
            f"{100.0 * tol:.1f}%). Try `data.tdi: 1st`, or `data.tdi: off`; "
            f"`data.psd_notch_strict: false` proceeds regardless.")
        if self.psd_notch_strict:
            raise PSDNullError(msg)
        logger.warning("psd notch UNSTABLE (psd_notch_strict=false): %s", msg)

    def _fd_settings(self, td_settings):
        return self._lt.FDSettings(
            td_settings.N // 2 + 1, 1.0 / (td_settings.N * td_settings.dt),
            force_backend=td_settings.force_backend)

    def _produce_data_residual_array(self, params=None):
        if params is None:
            params = self.injection_parameters

        _channel_strain = self.waveform_generator(*self._get_params(params))

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

    def emri_injection_generator(self):
        self._data_residual_array = self._produce_data_residual_array()
        self.sensetivity_matrix = self._lt.SensitivityMatrix(
            self._data_residual_array.settings,
            self.sensetivity_list,
            **noise_sens_kwargs(self.duration, self.foreground),
        )
        self._apply_psd_notch()
        self._analysis_container = self._lt.AnalysisContainer(
            self._data_residual_array,
            self.sensetivity_matrix,
            signal_gen=self.waveform_generator,
        )

        # rescale distance to get injection SNR
        if self.injection_snr is not None:
            self.injection_parameters["luminosity_distance"] /= (
                self.injection_snr / self._analysis_container.snr()
            )

            self.data_residual_array = self._produce_data_residual_array()

            assert np.all(
                self.data_residual_array.settings.f_arr
                == self._data_residual_array.settings.f_arr
            )

            self.analysis_container = self._lt.AnalysisContainer(
                self.data_residual_array,
                self.sensetivity_matrix,
                signal_gen=self.waveform_generator,
            )
            assert np.isclose(self.injection_snr, self.analysis_container.snr())
        else:
            self.data_residual_array = self._data_residual_array
            self.analysis_container = self._analysis_container

        self.optimal_snr = float(np.real(self.analysis_container.snr()))

        if self.add_noise:
            self._add_noise_realization()

    def _add_noise_realization(self):
        """Add a Gaussian noise realization drawn from the sensitivity PSD to the
        frequency-domain data.

        Convention (validated against lisatools.inner_product): per good frequency
        bin, n_tilde = sqrt(S(f) * T / 4) * (x + i y) with x, y ~ N(0,1), giving
        E[<n|n>] = 2 per bin per channel (chi^2 dof). Bins where the PSD is
        non-finite or non-positive (DC) get no noise.
        """
        fd = self.data_residual_array.data_res_arr  # FDSignal
        xp = fd.xp
        _sens_mat = self.sensetivity_matrix.sens_mat
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
            self.sensetivity_matrix,
            signal_gen=self.waveform_generator,
        )
        logger.info("noise: added PSD realization (seed=%s, %d/%d bins)",
                    self.noise_seed, ngood, nf)

    def generate_signal(self, params):
        return self._produce_data_residual_array(params=params)

    def evaluate_likelihood(self, input, full=None):
        """Log-likelihood for a template.

        By default (full=None -> self.full_likelihood, i.e. False) returns only the
        template-varying part <d|h> - 1/2<h|h>, dropping the constant noise and
        <d|d> terms. 
        Pass full=True (or construct with full_likelihood=True) to get the absolute,
        correctly-normalised ln L = noise + (-1/2)(<d|d> + <h|h> - 2<d|h>).
        """
        if full is None:
            full = self.full_likelihood

        if isinstance(input, dict):
            sig_dat_array = self.generate_signal(input)
        elif isinstance(input, self._lt.DataResidualArray):
            sig_dat_array = input
        else:
            raise TypeError("please input a dict of parameters or a DataResidualArray")

        d_h = self._lt.inner_product(
            self.data_residual_array, sig_dat_array,
            psd=self.sensetivity_matrix, normalize=False,
        )
        h_h = self._lt.inner_product(
            sig_dat_array, sig_dat_array,
            psd=self.sensetivity_matrix, normalize=False,
        )
        varying_term = d_h - 0.5 * h_h
        if full:
            # Full ln L = noise + (-1/2)(<d|d> + <h|h> - 2<d|h>)
            #           = (_noise_term - 1/2 <d|d>) + (<d|h> - 1/2 <h|h>).
            return self._lnlike_const + varying_term
        return varying_term


class LisatoolsEMRILikelihood(EMRIInjectionGenerator, InjectionModel):
    def __init__(self, injection_parameters, vectorized=False, **kwargs):
        super().__init__(injection_parameters, **kwargs)
        self.vectorized = vectorized
        self.default_parameters = {
            "mass_1": 1e6,
            "mass_2": 10.0,
            "a": 0.0,
            "p": 10.0,
            "e": 0.1,
            "x": 1.0,
            "q_k": 1.0,
            "phi_k": 0.0,
            "q_s": 1.0,
            "phi_s": 0.0,
            "luminosity_distance": 1.0,
            "phi_phi": 0.0,
            "phi_theta": 0.0,
            "phi_r": 0.0,
        }

    @classmethod
    def from_config(cls, cfg):
        _channels = getattr(cfg.data, "channels", None)
        return cls(
            dict(cfg.injection), vectorized=False,
            duration=cfg.data.duration, delta_t=cfg.data.delta_t,
            injection_snr=cfg.data.inj_snr,
            channel_list=None if _channels is None else list(_channels),
            tdi=str(getattr(cfg.data, "tdi", "2nd generation")),
            foreground=bool(getattr(cfg.data, "foreground", True)),
            add_noise=bool(getattr(cfg.data, "add_noise", False)),
            noise_seed=int(getattr(cfg.data, "noise_seed", 0)),
            pad_fft=bool(getattr(cfg.data, "pad_fft", True)),
            psd_notch=float(getattr(cfg.data, "psd_notch", 1e-5)),
            psd_notch_depth=float(getattr(cfg.data, "psd_notch_depth", 2.0)),
            psd_notch_strict=bool(getattr(cfg.data, "psd_notch_strict", True)),
        )

    def __call__(self, params):
        try:
            log_mass_1 = params[0]
            log_mass_2 = params[1]
            mass_1, mass_2 = mass1_mass2_from_log_masses(log_mass_1, log_mass_2)
            spin = params[2]
            semilatus_rectum = params[3]
            eccentricity = params[4]
            dist = params[5]
            q_s = params[6]
            phi_s = params[7]
            q_k = params[8]
            phi_k = params[9]
            phi_phi = params[10]
            phi_r = params[11]
        except Exception:
            return -np.inf

        template_params = self.default_parameters.copy()
        template_params["mass_1"] = mass_1
        template_params["mass_2"] = mass_2
        template_params["a"] = spin
        template_params["p"] = semilatus_rectum
        template_params["e"] = eccentricity
        template_params["luminosity_distance"] = dist
        template_params["q_s"] = q_s
        template_params["phi_s"] = phi_s
        template_params["q_k"] = q_k
        template_params["phi_k"] = phi_k
        template_params["phi_phi"] = phi_phi
        template_params["phi_r"] = phi_r

        try:
            result = self.evaluate_likelihood(template_params)

            if hasattr(result, "get"):
                result = result.get()
            result = np.real(result)
        except Exception:
            return -np.inf

        if np.isfinite(result):
            return float(result)
        else:
            return -np.inf
