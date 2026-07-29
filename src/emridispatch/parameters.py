"""Shared parameter-space definitions for the 12-D EMRI sampling vector.

Sampling vector (masses in log): [ln m1, ln m2, a, p, e, dist, q_s,
phi_s, q_k, phi_k, phi_phi, phi_r]. x and phi_theta are not sampled
(equatorial model -> Fisher-singular).
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

# Sampling-vector row -> injection-dict key; rows 0/1 hold the log of theirs.
VECTOR_TO_PHYSICAL = {
    "ln_m1": "mass_1", "ln_m2": "mass_2", "a": "a", "p": "p", "e": "e",
    "dist": "luminosity_distance", "q_s": "q_s", "phi_s": "phi_s",
    "q_k": "q_k", "phi_k": "phi_k", "phi_phi": "phi_phi", "phi_r": "phi_r",
}

# Fisher covariance / prior-box intrinsic parameter order (injection-dict keys).
INTRINSIC_ORDER = ["mass_1", "mass_2", "a", "p", "e", "luminosity_distance"]
# Sampled in log space (rows 0/1 of the sampling vector).
LOG_PARAMS = ("mass_1", "mass_2")

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


def mass1_mass2_from_log_masses(log_mass_1, log_mass_2):
    return np.exp(log_mass_1), np.exp(log_mass_2)


def truth_vector(inj):
    """The 12-D sampling vector at the injection (masses in log)."""
    vec = np.array([float(inj[VECTOR_TO_PHYSICAL[name]]) for name in PARAM_NAMES])
    vec[0], vec[1] = np.log(vec[0]), np.log(vec[1])
    return vec


def physical_from_vector(vec, fiducial):
    """Injection dict for a 12-D sampling vector, over a fiducial dict.

    Inverse of truth_vector; both read VECTOR_TO_PHYSICAL. Parameters the
    vector does not carry (x, phi_theta) keep their fiducial values.
    """
    out = dict(fiducial)
    for i, name in enumerate(PARAM_NAMES):
        out[VECTOR_TO_PHYSICAL[name]] = float(vec[i])
    m1, m2 = mass1_mass2_from_log_masses(out["mass_1"], out["mass_2"])
    out["mass_1"], out["mass_2"] = float(m1), float(m2)
    return out
