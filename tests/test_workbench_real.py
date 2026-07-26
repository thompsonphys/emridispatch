"""Real-stack checks: the unit suite runs against a stub, which cannot catch a
mismatch with the actual lisatools container shape."""

import os

import pytest

pytest.importorskip("few")
pytest.importorskip("lisatools.response")

CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples", "eryn_config.yaml")


def test_injection_template_round_trips_on_real_stack():
    from emridispatch.workbench import (
        injection_template, load, measure, overlap, snr)

    _cfg, model = load(CONFIG)
    h = injection_template(model)

    assert snr(model, h).optimal == pytest.approx(model.optimal_snr, rel=1e-6)
    assert overlap(model, h) == pytest.approx(1.0, abs=1e-6)

    result = measure(model, h, per_channel=True)
    per_chan_sq = sum(v.optimal ** 2 for v in result.snr.per_channel.values())
    assert per_chan_sq == pytest.approx(result.snr.optimal ** 2, rel=1e-8)
