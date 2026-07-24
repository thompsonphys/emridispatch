import numpy as np
import pytest
from scipy.integrate import quad

from emridispatch.priors import (
    CallablePrior, Cosine, Gaussian, JointPrior, LogUniform, PeriodicUniform,
    Sine, Uniform, joint_prior_from_box, prior_from_spec,
)


@pytest.mark.parametrize("prior", [
    Uniform(-1.0, 3.0),
    PeriodicUniform(0.0, 2 * np.pi),
    LogUniform(0.5, 20.0),
    Gaussian(1.0, 0.5),
    Gaussian(1.0, 0.5, minimum=0.0, maximum=2.0),
    Sine(),
    Cosine(),
])
def test_normalization(prior):
    lo = prior.minimum if np.isfinite(prior.minimum) else prior.mu - 20 * prior.sigma
    hi = prior.maximum if np.isfinite(prior.maximum) else prior.mu + 20 * prior.sigma
    total, _ = quad(lambda x: np.exp(prior.log_prob(x)), lo, hi, limit=200)
    assert np.isclose(total, 1.0, atol=1e-6)


@pytest.mark.parametrize("prior", [
    Uniform(-1.0, 3.0),
    LogUniform(0.5, 20.0),
    Gaussian(1.0, 0.5, minimum=0.0, maximum=2.0),
    Sine(),
    Cosine(),
])
def test_samples_in_bounds(prior):
    rng = np.random.default_rng(0)
    x = prior.sample(rng, size=5000)
    assert np.all(x >= prior.minimum) and np.all(x <= prior.maximum)
    # And log_prob is finite there.
    assert np.all(np.isfinite(prior.log_prob(x)))


def test_out_of_bounds_is_minus_inf():
    assert Uniform(0.0, 1.0).log_prob(1.5) == -np.inf
    assert LogUniform(1.0, 2.0).log_prob(0.5) == -np.inf
    assert Sine().log_prob(-0.1) == -np.inf


def test_callable_prior():
    p = CallablePrior(lambda x: -x, 0.0, 5.0)
    assert np.isclose(p.log_prob(2.0), -2.0)
    assert p.log_prob(6.0) == -np.inf
    rng = np.random.default_rng(0)
    assert 0.0 <= p.sample(rng) <= 5.0


def test_prior_from_spec():
    p = prior_from_spec({"type": "loguniform", "min": 0.5, "max": 20.0})
    assert isinstance(p, LogUniform)
    p = prior_from_spec({"type": "sine"})
    assert isinstance(p, Sine)
    p = prior_from_spec({"type": "gaussian", "mu": 1.0, "sigma": 0.1})
    assert isinstance(p, Gaussian)
    # Defaults from the box.
    p = prior_from_spec({"type": "uniform"}, default_min=-1.0, default_max=2.0)
    assert p.minimum == -1.0 and p.maximum == 2.0
    with pytest.raises(ValueError):
        prior_from_spec({"type": "nope"})
    with pytest.raises(ValueError):
        prior_from_spec({"type": "uniform", "min": 0, "max": 1, "junk": 2})


class TestJointPrior:
    """Semantics ported from the ancestral LnPriorUniformWvfm."""

    def setup_method(self):
        self.mins = np.array([0.0, 0.0, 0.0])
        self.maxes = np.array([1.0, 2 * np.pi, 5.0])
        self.jp = joint_prior_from_box(self.mins, self.maxes, periodic_indices=[1])

    def test_inside_constant(self):
        v1 = self.jp([0.5, 1.0, 2.0])
        v2 = self.jp([0.2, 3.0, 4.9])
        assert np.isfinite(v1) and np.isclose(v1, v2)

    def test_outside_minus_inf(self):
        assert self.jp([1.5, 1.0, 2.0]) == -np.inf
        assert self.jp([0.5, 1.0, -0.1]) == -np.inf

    def test_periodic_wrap_before_bounds_check(self):
        # Index 1 is periodic: 2*pi + 1 wraps to 1 -> in bounds.
        inside = self.jp([0.5, 1.0, 2.0])
        wrapped = self.jp([0.5, 2 * np.pi + 1.0, 2.0])
        assert np.isclose(inside, wrapped)
        # Non-periodic index does NOT wrap.
        assert self.jp([1.5, 1.0, 2.0]) == -np.inf

    def test_nan_is_minus_inf(self):
        assert self.jp([np.nan, 1.0, 2.0]) == -np.inf
        assert self.jp([0.5, np.nan, 2.0]) == -np.inf

    def test_batch(self):
        batch = np.array([[0.5, 1.0, 2.0], [1.5, 1.0, 2.0]])
        out = self.jp(batch)
        assert out.shape == (2,)
        assert np.isfinite(out[0]) and out[1] == -np.inf

    def test_periodic_metadata(self):
        assert self.jp.periodic == {1: 2 * np.pi}

    def test_initial_sample_in_bounds(self):
        x = self.jp.initial_sample()
        assert np.all(x >= self.mins) and np.all(x <= self.maxes)

    def test_mins_maxes(self):
        assert np.allclose(self.jp.mins, self.mins)
        assert np.allclose(self.jp.maxes, self.maxes)


def test_joint_prior_overrides():
    mins = np.array([0.5, 0.0])
    maxes = np.array([20.0, np.pi])
    jp = joint_prior_from_box(
        mins, maxes, names=["dist", "q_s"],
        overrides={"dist": {"type": "loguniform"}, "q_s": {"type": "sine"}})
    assert isinstance(jp["dist"], LogUniform)
    assert isinstance(jp["q_s"], Sine)
    # Bounds inherited from the box.
    assert jp["dist"].minimum == 0.5 and jp["dist"].maximum == 20.0
    # Non-uniform: log-prob varies with position now.
    assert not np.isclose(jp([1.0, 1.0]), jp([10.0, 1.0]))


def test_joint_prior_override_unknown_name():
    with pytest.raises(ValueError):
        joint_prior_from_box([0.0], [1.0], names=["a"], overrides={"b": {}})
