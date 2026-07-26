"""StableEMRIFisher provider (optional: `pip install emridispatch[fisher]`).

Needs stableemrifisher, few and lisatools/fastlisaresponse only at
compute() time; the module itself imports clean.
"""

import logging

import numpy as np

from emridispatch.fisher import FisherResult
from emridispatch.noise import (
    channel_noise_psd, per_channel_noise_kwargs, sensitivity_spec)

logger = logging.getLogger(__name__)


def _gpu_available():
    """True iff a CUDA device is actually usable (not just cupy importable)."""
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


class SEFFisherProvider:
    name = "sef"

    def __init__(self, tdi="2nd generation", foreground=True, channels=None):
        self.tdi = tdi
        self.foreground = foreground
        self.channels = channels

    def compute(self, injection_parameters, *, duration, delta_t, use_gpu=None):
        sigmas, cov, order = get_parameter_precision(
            injection_parameters, duration=duration, delta_t=delta_t,
            use_gpu=use_gpu, tdi=self.tdi, foreground=self.foreground,
            channels=self.channels)
        return FisherResult(sigmas=sigmas, cov=cov, order=order)


def get_parameter_precision(input_parameters, duration=0.01, delta_t=5.0,
                            use_gpu=None, tdi="2nd generation",
                            foreground=True, channels=None):
    if use_gpu is None:
        use_gpu = _gpu_available()
    force_backend = None if use_gpu else "cpu"

    if use_gpu:
        try:
            import cupy

            cupy.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    try:
        from few.waveform import (
            GenerateEMRIWaveform,
            FastKerrEccentricEquatorialFlux,
        )
        from stableemrifisher.fisher import StableEMRIFisher
    except ImportError as err:
        raise ImportError(
            "StableEMRIFisher / FastEMRIWaveforms are required for the 'sef' "
            "Fisher provider. Install with `pip install emridispatch[fisher]` (plus "
            "the lisatools extra), or configure prior.fisher: manual with your "
            "own sigmas/covariance."
        ) from err

    # Waveform params -- match the injection's observation (duration/cadence) so
    # the Fisher-derived widths correspond to the SAME data the sampler analyses.
    dt = delta_t
    T = duration
    wave_params = {
        "m1": input_parameters["mass_1"],
        "m2": input_parameters["mass_2"],
        "a": input_parameters["a"],
        "p0": input_parameters["p"],
        "e0": input_parameters["e"],
        "xI0": input_parameters["x"],
        "dist": input_parameters["luminosity_distance"],
        "qS": input_parameters["q_s"],
        "phiS": input_parameters["phi_s"],
        "qK": input_parameters["q_k"],
        "phiK": input_parameters["phi_k"],
        "Phi_phi0": input_parameters["phi_phi"],
        "Phi_theta0": input_parameters["phi_theta"],
        "Phi_r0": input_parameters["phi_r"],
    }

    waveform_class = FastKerrEccentricEquatorialFlux
    waveform_class_kwargs = {
        "inspiral_kwargs": {
            "err": 1e-11,
        },
        "mode_selector_kwargs": {"mode_selection_threshold": 1e-5},
        "sum_kwargs": {"pad_output": True},
        "force_backend": force_backend,  # "cpu" or None (GPU auto)
    }
    waveform_generator = GenerateEMRIWaveform
    waveform_generator_kwargs = {"return_list": False, "frame": "detector"}

    INDEX_LAMBDA = 8
    INDEX_BETA = 7

    # with longer signals we care less about this
    t0 = 20000.0  # throw away on both ends when our orbital information is weird

    sens_classes, channels = sensitivity_spec(tdi, channels)
    extra_sef_kwargs = dict(
        noise_model=channel_noise_psd,
        noise_kwargs=per_channel_noise_kwargs(T, foreground, sens_classes),
        channels=channels,
    )

    if tdi == "off":
        ResponseWrapper = None
        ResponseWrapper_kwargs = None
        logger.info("fisher: tdi off (no ResponseWrapper, channels %s, "
                    "foreground=%s)", channels, foreground)
    else:
        try:
            from lisatools.response import ResponseWrapper
        except ModuleNotFoundError:
            from fastlisaresponse import ResponseWrapper
        from lisatools.detector import EqualArmlengthOrbits

        tdi_kwargs = dict(
            orbits=EqualArmlengthOrbits(force_backend=force_backend),
            order=25,
            tdi=tdi,
            tdi_chan="".join(channels),
        )
        ResponseWrapper_kwargs = dict(
            Tobs=T,
            dt=dt,
            index_lambda=INDEX_LAMBDA,
            index_beta=INDEX_BETA,
            t0=t0,
            flip_hx=True,
            force_backend=force_backend,
            is_ecliptic_latitude=False,
            remove_garbage="zero",
            **tdi_kwargs,
        )
        logger.info("fisher: tdi %s (channels %s, foreground=%s)",
                    tdi, channels, foreground)

    der_order = 4
    Ndelta = 8
    sef = StableEMRIFisher(
        waveform_class=waveform_class,
        waveform_class_kwargs=waveform_class_kwargs,
        waveform_generator=waveform_generator,
        waveform_generator_kwargs=waveform_generator_kwargs,
        ResponseWrapper=ResponseWrapper,
        ResponseWrapper_kwargs=ResponseWrapper_kwargs,
        stats_for_nerds=True,
        use_gpu=use_gpu,
        T=T,
        dt=dt,
        der_order=der_order,
        Ndelta=Ndelta,
        stability_plot=False,
        return_derivatives=False,
        deriv_type="stable",
        **extra_sef_kwargs,
    )

    param_names = ["m1", "m2", "a", "p0", "e0", "dist"]

    delta_range = dict(
        m1=np.geomspace(1e3, 1e-5, Ndelta),
        m2=np.geomspace(1e-2, 1e-8, Ndelta),
        a=np.geomspace(1e-5, 1e-9, Ndelta),
        p0=np.geomspace(1e-5, 1e-9, Ndelta),
        e0=np.geomspace(1e-5, 1e-9, Ndelta),
        dist=np.geomspace(1e-5, 1e-9, Ndelta),
    )

    fisher_matrix = sef(
        wave_params,
        param_names=param_names,
        delta_range=delta_range,
        filename=None,
        live_dangerously=False,
    )

    param_cov = np.linalg.inv(fisher_matrix)
    param_cov = 0.5 * (param_cov + param_cov.T)
    key_map = {
        "m1": "mass_1",
        "m2": "mass_2",
        "a": "a",
        "p0": "p",
        "e0": "e",
        "dist": "luminosity_distance",
    }

    order = [key_map[k] for k in delta_range.keys()]
    sigmas = {name: param_cov[i, i] ** (1 / 2) for i, name in enumerate(order)}

    return sigmas, param_cov, order
