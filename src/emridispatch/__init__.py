"""emridispatch: EMRI parameter estimation with pluggable samplers, TDI backends,
Fisher providers, and flexible priors.

Core needs only numpy/scipy/pyyaml; optional extras: impulse (sampler
backend), lisatools (response), fisher (StableEMRIFisher sizing), gmm
(mode-jump proposal), diagnostics (chain diagnostics).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
