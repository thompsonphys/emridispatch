"""Impulse backend: pure-numpy builder unit tests.

build_ladder and build_mode_jumps depend only on numpy, so this file runs
without impulse installed.
"""

import types

import numpy as np
import pytest

from emridispatch.backends.impulse.ladder import build_ladder
from emridispatch.backends.impulse.mode_jumps import build_mode_jumps


def ladder_cfg(**kw):
    return types.SimpleNamespace(**{"max_temp": 1000.0, "t_split": 25.0,
                                    "ntemps_low": 20, "ntemps_high": 6, **kw})


def test_ladder_geometry():
    t = build_ladder(ladder_cfg())
    assert len(t) == 27
    assert np.all(np.diff(t) > 0)
    assert (t[0], t[19], t[-2]) == (1.0, 25.0, 1000.0)
    assert np.isinf(t[-1]) and np.count_nonzero(np.isinf(t)) == 1
    # geomspace, not linspace: the 3-rung low block bisects [1, 25] in log.
    assert np.allclose(build_ladder(ladder_cfg(ntemps_low=3, ntemps_high=2))[:3],
                       [1.0, 5.0, 25.0])


@pytest.mark.parametrize("kw", [
    {"ntemps_low": 0},          # drops the T=1 cold rung
    {"t_split": 5000.0},        # t_split above max_temp -> decreasing
    {"t_split": 1.0},           # duplicate rungs at T=1
    {"max_temp": 25.0},         # max_temp == t_split -> duplicate rungs
])
def test_ladder_rejects_broken_geometry(kw):
    with pytest.raises(ValueError, match="strictly increase"):
        build_ladder(ladder_cfg(**kw))


@pytest.mark.parametrize("kw", [
    {"ntemps_high": 0}, {"ntemps_low": 1}, {"t_split": 1.0, "ntemps_low": 1},
])
def test_ladder_accepts_degenerate_but_valid_geometry(kw):
    t = build_ladder(ladder_cfg(**kw))
    assert t[0] == 1.0 and np.all(np.diff(t) > 0)


@pytest.mark.parametrize("method,names", [
    ("none", []),
    ("popde", ["popde"]),
    ("gmm", ["gmm_mode"]),
    ("popde+gmm", ["popde", "gmm_mode"]),
])
def test_mode_jump_methods(method, names):
    jumps = build_mode_jumps(method, 14, np.arange(6), weight=7.0)
    assert [j.__name__ for j, _ in jumps] == names
    assert [w for _, w in jumps] == [7.0] * len(names)
    # One pool shared across every jump: hot chains feed the cold one.
    assert len({id(j.pool) for j, _ in jumps}) == min(len(names), 1)


def test_pool_wraps_at_capacity():
    from emridispatch.backends.impulse.mode_jumps import CrossChainPool

    pool = CrossChainPool(2, capacity=4)
    for i in range(3):
        pool.push(np.full(2, float(i)))
    assert len(pool) == 3 and pool.view().shape == (3, 2)
    for i in range(3, 9):
        pool.push(np.full(2, float(i)))
    assert len(pool) == 4 and pool.view().shape == (4, 2)
    # Ring buffer: the four most recent pushes survive, oldest overwritten.
    assert sorted(pool.view()[:, 0]) == [5.0, 6.0, 7.0, 8.0]


@pytest.mark.parametrize("method", ["POPDE", "GMM", "nogmm", "typo", "", None, False])
def test_unknown_mode_jump_method_raises(method):
    with pytest.raises(ValueError, match="mode_jump.method"):
        build_mode_jumps(method, 14, np.arange(6))
