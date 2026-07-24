"""Shared parameter-space definitions for the 12-D EMRI sampling vector.

Sampling-vector order (masses in log), matching the likelihood and diagnostics:
    [ln m1, ln m2, a, p, e, dist, q_s, phi_s, q_k, phi_k, phi_phi, phi_r]
The first six are intrinsic/distance; the last six are sky/spin angles + initial
phases. Inclination x and polar phase phi_theta are not sampled (the
FastKerrEccentricEquatorial model is equatorial -> Fisher-singular).
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
    return np.array([
        np.log(inj["mass_1"]), np.log(inj["mass_2"]),
        inj["a"], inj["p"], inj["e"], inj["luminosity_distance"],
        inj["q_s"], inj["phi_s"], inj["q_k"], inj["phi_k"], inj["phi_phi"], inj["phi_r"],
    ])
