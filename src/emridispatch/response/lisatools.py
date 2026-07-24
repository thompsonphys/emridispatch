"""EMRI injection + matched-filter likelihood via lisa-analysis-tools + FEW.

Optional dependency: `pip install emridispatch[lisatools]`. All lisatools imports
happen at construction time, so this module imports clean without it.

Ported from gwsampling's emri/injection_generator.py + emri/likelihood.py, with
the Fisher computation excised (see emridispatch.fisher for providers).
"""

import logging
from types import SimpleNamespace

import numpy as np

from emridispatch.parameters import mass1_mass2_from_log_masses
from emridispatch.response import InjectionModel

logger = logging.getLogger(__name__)


def _import_lisatools():
    try:
        from lisatools.datacontainer import DataResidualArray
        from lisatools.domains import TDSettings
        from lisatools.sensitivity import (
            SensitivityMatrix, A1TDISens, E1TDISens, T1TDISens)
        from lisatools.analysiscontainer import AnalysisContainer
        from lisatools.diagnostic import inner_product, noise_likelihood_term
        from lisatools.sources.emri import EMRITDIWaveform
        from lisatools.utils.constants import YRSID_SI
    except ImportError as err:
        raise ImportError(
            "lisa-analysis-tools (and FastEMRIWaveforms) are required for the "
            "'lisatools' response model. Install with "
            "`pip install emridispatch[lisatools]`."
        ) from err
    return SimpleNamespace(
        DataResidualArray=DataResidualArray,
        TDSettings=TDSettings,
        SensitivityMatrix=SensitivityMatrix,
        sens_by_channel={"A": A1TDISens, "E": E1TDISens, "T": T1TDISens},
        AnalysisContainer=AnalysisContainer,
        inner_product=inner_product,
        noise_likelihood_term=noise_likelihood_term,
        EMRITDIWaveform=EMRITDIWaveform,
        YRSID_SI=YRSID_SI,
    )


class EMRIInjectionGenerator:
    def __init__(
        self,
        injection_parameters,
        duration=1,
        delta_t=10,
        injection_snr=None,
        channel_list=None,
        full_likelihood=False,
        add_noise=False,
        noise_seed=0,
    ):
        self._lt = _import_lisatools()

        self.injection_parameters = injection_parameters
        self.injection_snr = injection_snr
        self.delta_t = delta_t
        self.duration = duration
        # Optional Gaussian noise realization added to the data (drawn from the
        # sensitivity PSD). noise_seed is a DEDICATED seed -- independent of the
        # sampler's run.seed, so noise draws and chain randomness decouple.
        self.add_noise = add_noise
        self.noise_seed = noise_seed
        # When False (default), evaluate_likelihood returns only the
        # template-VARYING part of ln L (drops the constant noise + <d|d> terms).
        # Those constants (~1e8) swamp the ~1e2 signal contrast in floating point,
        # so dropping them recovers precision -- and a constant offset cancels in
        # the MCMC acceptance ratio and PT swaps, so the posterior is unchanged.
        # Set True only when the absolute, correctly-normalised ln L is needed
        # (e.g. evidence / thermodynamic integration).
        self.full_likelihood = full_likelihood

        if channel_list is None:
            self.channel_list = ["A", "E"]
        else:
            self.channel_list = channel_list

        try:
            self.sensetivity_list = [
                self._lt.sens_by_channel[CS] for CS in self.channel_list]
        except KeyError as err:
            raise ValueError(
                f"unknown TDI channel {err.args[0]!r}; choose from "
                f"{sorted(self._lt.sens_by_channel)}") from err

        self.channel_string = "".join(self.channel_list)

        self.waveform_generator = self._lt.EMRITDIWaveform(
            response_kwargs=dict(
                t0=30000.0, tdi="1st generation", tdi_chan=self.channel_string
            ),
            T=self.duration,
            dt=self.delta_t,
        )

        # Build the injection and calibrate the distance to hit injection_snr.
        self.emri_injection_generator()

        # Two pieces of the likelihood are CONSTANT for the whole run (they depend
        # only on the fixed data + sensitivity matrix, not the template), so compute
        # them once here instead of on every likelihood call:
        #   * the noise term          -sum log|detC|
        #   * the data auto-inner-product  <d|d>
        # Only <d|h> and <h|h> vary per template (see evaluate_likelihood).
        self._noise_term = self._lt.noise_likelihood_term(self.sensetivity_matrix)
        self._d_d = self._lt.inner_product(
            self.data_residual_array,
            self.data_residual_array,
            psd=self.sensetivity_matrix,
            normalize=False,
        )
        # Constant part of ln L = noise term + (-1/2 <d|d>). Added back only when
        # full_likelihood is requested; otherwise it is dropped for precision.
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

    def _produce_data_residual_array(self, params=None):
        if params is None:
            params = self.injection_parameters

        _channel_strain = self.waveform_generator(*self._get_params(params))
        # Keep the TDI channels on whatever device the waveform generator used
        # (GPU/cupy when available) so the whole data path -- stacking, FFT and
        # inner products -- stays on the GPU. Only fall back to host/numpy when
        # cupy is not installed (e.g. a CPU-only machine).
        try:
            import cupy as _cp

            xp = _cp.get_array_module(_channel_strain[0])
        except ImportError:
            xp = np
        channel_strain = xp.asarray([xp.asarray(AET) for AET in _channel_strain])
        td_settings = self._lt.TDSettings(channel_strain.shape[-1], self.delta_t)
        return self._lt.DataResidualArray(
            channel_strain, input_signal_domain=td_settings)

    def emri_injection_generator(self):
        self._data_residual_array = self._produce_data_residual_array()
        self.sensetivity_matrix = self._lt.SensitivityMatrix(
            self._data_residual_array.settings,
            self.sensetivity_list,
            stochastic_params=(self.duration * self._lt.YRSID_SI,),
        )
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

        # Optimal (signal-only) SNR of the injection, recorded BEFORE any noise is
        # added: with noise in the data, analysis_container.snr() = sqrt(<d|d>)
        # includes the noise power and is no longer the injected signal SNR.
        self.optimal_snr = float(np.real(self.analysis_container.snr()))

        # Noise LAST: SNR calibration above must run against the clean signal
        # (its snr() assertion would fail on noisy data). The <d|d> / noise-term
        # constants are computed in __init__ AFTER this returns, so they pick up
        # the noisy data automatically.
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
        S = np.asarray(self.sensetivity_matrix.sens_mat)   # (nchan, nf), host
        nchan, nf = S.shape
        # Observation time from the actual frequency resolution (the generated
        # array's length can differ slightly from duration*YRSID_SI/delta_t).
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
        template-VARYING part <d|h> - 1/2<h|h>, dropping the constant noise and
        <d|d> terms. Those constants are ~1e8 while the signal contrast is ~1e2, so
        keeping them costs floating-point precision; dropping a constant offset does
        not change the MCMC posterior (it cancels in acceptance ratios / PT swaps).
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

        # Only <d|h> and <h|h> depend on the template; <d|d> and the noise term are
        # cached constants. The varying part is <d|h> - 1/2<h|h> (this is what the
        # sampler needs, at full precision).
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


# Gaussian matched-filter likelihood <d|h> - 1/2<h|h> (constants dropped).
# Data is the injected signal, optionally plus a PSD noise realization
# (add_noise/noise_seed kwargs, handled by EMRIInjectionGenerator).
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
        return cls(
            dict(cfg.injection), vectorized=False,
            duration=cfg.data.duration, delta_t=cfg.data.delta_t,
            injection_snr=cfg.data.inj_snr,
            channel_list=list(cfg.data.channels),
            add_noise=bool(getattr(cfg.data, "add_noise", False)),
            noise_seed=int(getattr(cfg.data, "noise_seed", 0)),
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
            # Full-space angles/phases (equatorial Kerr): sky location (q_s, phi_s),
            # spin orientation (q_k, phi_k) and the radial/azimuthal initial phases
            # (phi_phi, phi_r). x (inclination) and phi_theta (polar phase) stay at
            # their injected values -- the FastKerrEccentricEquatorial model is
            # equatorial, so both are fixed, not sampled.
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

        # evaluate_likelihood returns the log-likelihood ln L. By default it is the
        # template-VARYING part only (<d|h> - 1/2<h|h>); the constant noise + <d|d>
        # terms are dropped for floating-point precision and cancel in the sampler
        # anyway (construct with full_likelihood=True for the absolute value).
        # The sampler consumes exp(lnprior + lnlike/T), so return +ln L directly --
        # do NOT take a second log or negate. (np.real: inner products can carry a
        # tiny imaginary part.) Any waveform failure (e.g. a proposal in an invalid
        # region of parameter space) => -inf so the sampler just rejects it.
        try:
            result = self.evaluate_likelihood(template_params)
            # evaluate_likelihood runs on the GPU, so the returned scalar is a
            # cupy 0-d array; pull the single value back to the host so the
            # sampler (which expects numpy/python floats) can consume it.
            if hasattr(result, "get"):
                result = result.get()
            result = np.real(result)
        except Exception:
            return -np.inf

        if np.isfinite(result):
            return float(result)
        else:
            return -np.inf
