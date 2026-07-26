"""Sampler-backend registry; imports are lazy so the core package never
requires a sampler.

A backend is a class with `name` and `run(problem, cfg, resume=True)
-> dict`; register externally via register_backend("name", "mod:Cls").
"""

import importlib

from emridispatch.backends.base import SamplerBackend, SamplingProblem  # noqa: F401

# name -> ("module.path:ClassName", pip-extra hint)
_REGISTRY = {
    "impulse": ("emridispatch.backends.impulse:ImpulseBackend", "emridispatch[impulse]"),
    "eryn": ("emridispatch.backends.eryn:ErynBackend", "emridispatch[eryn]"),
}


def register_backend(name, target, extra_hint=None):
    """Register a SamplerBackend implementation ("module.path:ClassName")."""
    _REGISTRY[str(name)] = (str(target), extra_hint)


def get_backend(name) -> SamplerBackend:
    name = str(name)
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown sampler backend {name!r}; registered: {sorted(_REGISTRY)}")
    target, extra = _REGISTRY[name]
    mod_path, cls_name = target.split(":")
    try:
        module = importlib.import_module(mod_path)
    except ImportError as err:
        hint = f" Install with `pip install {extra}`." if extra else ""
        raise ImportError(
            f"sampler backend {name!r} could not be imported ({err})." + hint
        ) from err
    return getattr(module, cls_name)()
