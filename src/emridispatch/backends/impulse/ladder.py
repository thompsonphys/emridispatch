"""Temperature-ladder construction for the EMRI PT-MCMC sampler.

Pure + testable (no sampler / GPU deps): the geometry is driven entirely by the
config's `sampler.impulse.ladder` section. See emri_config.yaml for the rationale (dense at low T
to bridge the mode-barrier melt zone, coarse above, inf chain as the top rung).
"""

import numpy as np


def build_ladder(cfg):
    """Dense-low-T temperature ladder with an explicit infinite top rung.

    Parameters
    ----------
    cfg : object with attributes
        max_temp, t_split, ntemps_low, ntemps_high
        (e.g. cfg.sampler.impulse.ladder).

    Returns
    -------
    numpy.ndarray
        `ntemps_low` rungs geometrically spaced in [1, t_split], `ntemps_high`
        rungs in (t_split, max_temp], then np.inf. Total = ntemps_low +
        ntemps_high + 1. The inf rung is the prior-only chain (beta = 0).
    """
    return np.concatenate(
        [
            np.geomspace(1.0, cfg.t_split, cfg.ntemps_low),
            np.geomspace(cfg.t_split, cfg.max_temp, cfg.ntemps_high + 1)[1:],
            [np.inf],
        ]
    )
