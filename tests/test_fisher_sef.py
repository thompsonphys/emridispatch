from types import SimpleNamespace

import emridispatch.fisher.sef as sef_mod
from emridispatch.fisher.sef import SEFFisherProvider


def test_provider_stores_and_forwards_tdi(monkeypatch):
    seen = {}

    def fake_gpp(params, *, duration, delta_t, use_gpu=None,
                 tdi="2nd generation"):
        seen["tdi"] = tdi
        sig = dict.fromkeys(
            ["mass_1", "mass_2", "a", "p", "e", "luminosity_distance"], 1.0)
        import numpy as np
        return sig, np.eye(6), list(sig)

    monkeypatch.setattr(sef_mod, "get_parameter_precision", fake_gpp)
    provider = SEFFisherProvider(tdi="off")
    provider.compute({}, duration=0.3, delta_t=10.0)
    assert seen["tdi"] == "off"


def test_provider_default_tdi():
    assert SEFFisherProvider().tdi == "2nd generation"


def test_get_fisher_provider_passes_tdi(monkeypatch):
    import emridispatch.fisher as fisher_mod

    monkeypatch.setattr(fisher_mod, "_sef_importable", lambda: True)
    cfg = SimpleNamespace(
        prior=SimpleNamespace(fisher="sef", sigmas=None, covariance_file=None),
        data=SimpleNamespace(tdi="off"))
    provider = fisher_mod.get_fisher_provider(cfg)
    assert provider.tdi == "off"
