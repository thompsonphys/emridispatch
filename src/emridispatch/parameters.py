"""Shared parameter-space definitions for the 12-D EMRI sampling vector.

PARAM_NAMES is the sampling vector (masses in log) and VECTOR_TO_PHYSICAL maps
its rows to injection-dict keys. Injection keys absent from that mapping are not
sampled at the moment.
"""

import numpy as np

PARAM_NAMES = [
    "ln_m1", "ln_m2", "a", "p", "e", "dist",
    "q_s", "phi_s", "q_k", "phi_k", "phi_phi", "phi_r",
]
NDIM = len(PARAM_NAMES)

# LaTeX axis labels, aligned with PARAM_NAMES (q_s/q_k are polar angles).
PARAM_LABELS = [
    r"$\ln m_1$", r"$\ln m_2$", r"$a$", r"$p_0$", r"$e_0$", r"$d_L$",
    r"$\theta_S$", r"$\phi_S$", r"$\theta_K$", r"$\phi_K$",
    r"$\Phi_\varphi$", r"$\Phi_r$",
]

# Sampling-vector row -> injection-dict key; LOG_ROWS hold the log of theirs.
VECTOR_TO_PHYSICAL = {
    "ln_m1": "mass_1", "ln_m2": "mass_2", "a": "a", "p": "p", "e": "e",
    "dist": "luminosity_distance", "q_s": "q_s", "phi_s": "phi_s",
    "q_k": "q_k", "phi_k": "phi_k", "phi_phi": "phi_phi", "phi_r": "phi_r",
}

# GenerateEMRIWaveform positional order -> injection-dict key. SEF keys its
# wave_params by these names and splats the values into the same generator.
FEW_TO_INJECTION = {
    "m1": "mass_1", "m2": "mass_2", "a": "a", "p0": "p", "e0": "e",
    "xI0": "x", "dist": "luminosity_distance", "qS": "q_s", "phiS": "phi_s",
    "qK": "q_k", "phiK": "phi_k", "Phi_phi0": "phi_phi",
    "Phi_theta0": "phi_theta", "Phi_r0": "phi_r",
}

# Fisher covariance / prior-box intrinsic parameter order (injection-dict keys).
INTRINSIC_ORDER = ["mass_1", "mass_2", "a", "p", "e", "luminosity_distance"]
# Sampled in log space. bounds.py and reparam.py additionally require these to
# be rows 0/1; truth_vector and physical_from_vector do not.
LOG_PARAMS = ("mass_1", "mass_2")
LOG_ROWS = tuple(i for i, name in enumerate(PARAM_NAMES)
                 if VECTOR_TO_PHYSICAL[name] in LOG_PARAMS)

# Full physical ranges for the six angle/phase rows (sampling-vector rows 6..11).
TWO_PI = 2.0 * np.pi
ANGLE_RANGES = [
    (0.0, np.pi),   # q_s   polar sky angle
    (0.0, TWO_PI),  # phi_s azimuthal sky angle (periodic)
    (0.0, np.pi),   # q_k   polar spin angle
    (0.0, TWO_PI),  # phi_k azimuthal spin angle (periodic)
    (0.0, TWO_PI),  # phi_phi azimuthal initial phase (periodic)
    (0.0, TWO_PI),  # phi_r   radial initial phase (periodic)
]

# Default 2*pi-periodic sampling-vector indices (phi_s, phi_k, phi_phi, phi_r).
DEFAULT_PERIODIC_2PI_INDICES = [7, 9, 10, 11]


def few_params(inj):
    """FEW-keyed parameters in GenerateEMRIWaveform/SEF positional order.

    """
    return {name: float(inj[key]) for name, key in FEW_TO_INJECTION.items()}


def truth_vector(inj):
    """The 12-D sampling vector at the injection (masses in log)."""
    vec = np.array([float(inj[VECTOR_TO_PHYSICAL[name]]) for name in PARAM_NAMES])
    vec[list(LOG_ROWS)] = np.log(vec[list(LOG_ROWS)])
    return vec


def physical_from_vector(vec, fiducial):
    """Injection dict for a 12-D sampling vector.

    Inverse of truth_vector; both read VECTOR_TO_PHYSICAL. Injection keys the
    vector does not carry keep their fiducial values, which load_config has
    already coerced to float.
    """
    out = dict(fiducial)
    for i, name in enumerate(PARAM_NAMES):
        out[VECTOR_TO_PHYSICAL[name]] = float(vec[i])
    for key in LOG_PARAMS:
        out[key] = float(np.exp(out[key]))
    return out
