import numpy as np
import pytest

from emridispatch.starts import initial_point

TRUTH = np.array([0.5, 0.5, 0.5])
COV = np.diag([0.01, 0.01, 0.01])
MINS = np.zeros(3)
MAXES = np.ones(3)


def test_truth_mode():
    x = initial_point("truth", TRUTH, COV, MINS, MAXES, seed=1)
    assert np.allclose(x, TRUTH)
    # Copy, not a view.
    x[0] = 99.0
    assert TRUTH[0] == 0.5


def test_prior_mode_in_box_and_seeded():
    x1 = initial_point("prior", TRUTH, COV, MINS, MAXES, seed=7)
    x2 = initial_point("prior", TRUTH, COV, MINS, MAXES, seed=7)
    x3 = initial_point("prior", TRUTH, COV, MINS, MAXES, seed=8)
    assert np.all(x1 >= MINS) and np.all(x1 <= MAXES)
    assert np.allclose(x1, x2)
    assert not np.allclose(x1, x3)


def test_fisher_mode_clipped():
    x = initial_point("fisher", TRUTH, COV, MINS, MAXES, seed=3, jitter=100.0)
    assert np.all(x >= MINS) and np.all(x <= MAXES)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="start_mode"):
        initial_point("nope", TRUTH, COV, MINS, MAXES, seed=1)
