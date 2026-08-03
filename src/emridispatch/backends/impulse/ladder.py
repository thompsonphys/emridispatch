"""Temperature-ladder construction for the EMRI PT-MCMC sampler.

Pure and GPU-independent; geometry is driven entirely by the config's
`sampler.impulse.ladder` section.
"""

import numpy as np


def build_ladder(cfg):
    """Dense-low-T temperature ladder with an explicit infinite top rung.

    Needs cfg.max_temp, t_split, ntemps_low, ntemps_high (e.g.
    cfg.sampler.impulse.ladder). Returns ntemps_low geomspaced rungs in
    [1, t_split], ntemps_high in (t_split, max_temp], then np.inf (beta=0).
    """
    ladder = np.concatenate(
        [
            np.geomspace(1.0, cfg.t_split, cfg.ntemps_low),
            np.geomspace(cfg.t_split, cfg.max_temp, cfg.ntemps_high + 1)[1:],
            [np.inf],
        ]
    )
    if ladder[0] != 1.0 or not np.all(np.diff(ladder) > 0):
        raise ValueError(
            f"temperature ladder must start at 1.0 and strictly increase, got "
            f"{ladder}; fix sampler.impulse.ladder")
    return ladder
