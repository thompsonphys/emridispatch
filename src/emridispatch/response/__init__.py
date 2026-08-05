"""Injection / likelihood models behind a pluggable TDI-response interface.

Register new models via register_model(name, "module:Class"); the class
needs a from_config(cfg) classmethod. Select with data.response: <name>.
"""

import importlib
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

__all__ = ["InjectionModel", "build_injection_model", "register_model"]


class InjectionModel(ABC):
    """Contract every response/likelihood implementation provides.

    injection_parameters: dict, post-SNR-calibration truth values.
    optimal_snr: float, the injection's optimal (signal-only) SNR.
    """

    injection_parameters: dict
    optimal_snr: float

    @abstractmethod
    def evaluate_likelihood(self, template_params) -> float:
        """ln L for a template given as a physical-parameter dict."""

    @abstractmethod
    def __call__(self, params) -> float:
        """ln L for the 12-D sampling vector (see emridispatch.parameters).

        Must return a plain float, -inf on any waveform failure.
        """


# name -> "module.path:ClassName". Values imported lazily so optional heavy
# dependencies (lisatools/FEW) are only touched when actually selected.
_REGISTRY = {
    "lisatools": "emridispatch.response.lisatools:LisatoolsEMRILikelihood",
    "toy": "emridispatch.response.toy:ToyGaussianLikelihood",
}


def register_model(name, target):
    """Register an InjectionModel implementation ("module.path:ClassName")."""
    _REGISTRY[str(name)] = str(target)


def build_injection_model(cfg):
    """Instantiate the model selected by cfg.data.response."""
    name = str(cfg.data.response)
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown data.response {name!r}; registered: {sorted(_REGISTRY)}")
    mod_path, cls_name = _REGISTRY[name].split(":")
    try:
        module = importlib.import_module(mod_path)
    except ImportError as err:
        hint = (" Install with `pip install emridispatch[lisatools]`."
                if name == "lisatools" else "")
        raise ImportError(
            f"response model {name!r} could not be imported ({err})." + hint
        ) from err
    cls = getattr(module, cls_name)
    logger.info("response: building %r injection model", name)
    return cls.from_config(cfg)
