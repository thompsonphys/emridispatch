"""emridispatch: EMRI parameter estimation with pluggable samplers, TDI backends,
Fisher providers, and flexible priors.

Core is numpy/scipy/pyyaml only; heavy machinery is optional extras:
    emridispatch[impulse]    - the impulse PT-MCMC sampler backend
    emridispatch[lisatools]  - lisa-analysis-tools + FastEMRIWaveforms response
    emridispatch[fisher]     - StableEMRIFisher prior-box / proposal sizing
    emridispatch[gmm]        - GMM mode-jump proposal (scikit-learn)
    emridispatch[diagnostics]- arviz/emcee/matplotlib chain diagnostics

Typical use:
    from emridispatch.config import load_config
    from emridispatch.pipeline import build_problem, run_from_config

    cfg = load_config("my_config.yaml")
    run_from_config(cfg)                    # full run via the configured backend
    problem = build_problem(cfg)            # or: sampler-agnostic problem only
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
