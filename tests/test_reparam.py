import numpy as np
import pytest

from emridispatch.reparam import GridReparam, Reparam, ReparamCallable

LM1, LM2, DIST = np.log(1.0e6), np.log(10.0), 1.0
GRID_COV = np.diag([1e-8, 1e-8, 1e-6, 1e-4, 1e-6, 1e-2])


def _separatrix(a, e):
    from few.utils.geodesic import get_separatrix

    return float(get_separatrix(np.array([abs(a)]), np.array([e]),
                                np.array([np.sign(a) or 1.0]))[0])


def _block(a, p, e):
    return np.array([LM1, LM2, a, p, e, DIST])


def _vector(a, p, e):
    return np.array([LM1, LM2, a, p, e, DIST, 0.7, 1.1, 0.9, 2.2, 0.3, 1.4])


def test_identity_roundtrip():
    rp = Reparam.identity(4)
    x = np.array([1.0, -2.0, 3.0, 0.5])
    assert np.allclose(rp.to_u(x), x)
    assert np.allclose(rp.to_x(x), x)
    assert rp.log_abs_det_jac(x) == 0.0


def test_from_covariance_roundtrip():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((3, 3))
    cov = A @ A.T + 0.1 * np.eye(3)
    mu = np.array([1.0, -1.0, 2.0])
    rp = Reparam.from_covariance(5, [0, 1, 2], cov, mu)

    x = rng.standard_normal(5)
    u = rp.to_u(x)
    assert np.allclose(rp.to_x(u), x)
    # Untouched dims pass through.
    assert np.allclose(u[3:], x[3:])


def test_whitening_decorrelates():
    rng = np.random.default_rng(7)
    A = rng.standard_normal((3, 3))
    cov = A @ A.T + 0.1 * np.eye(3)
    mu = np.zeros(3)
    rp = Reparam.from_covariance(3, [0, 1, 2], cov, mu)
    # Pushing the covariance through the map should give near-identity corr.
    cov_u = rp.transform_cov(cov)
    sig = np.sqrt(np.diag(cov_u))
    corr = cov_u / np.outer(sig, sig)
    assert np.allclose(corr, np.diag(np.diag(corr)), atol=1e-10)


def test_reparam_callable_wraps_and_jacobian():
    rp = Reparam.from_covariance(
        2, [0, 1], np.array([[4.0, 0.0], [0.0, 1.0]]), np.array([1.0, 2.0]))

    def lnf(x):
        return -np.sum(x ** 2)

    wrapped = ReparamCallable(lnf, rp)
    u = np.array([0.3, -0.7])
    assert np.isclose(wrapped(u), lnf(rp.to_x(u)))

    # Linear map: jacobian term is 0, so jacobian=True changes nothing.
    wrapped_j = ReparamCallable(lnf, rp, jacobian=True)
    assert np.isclose(wrapped_j(u), wrapped(u))


@pytest.mark.parametrize("a,e", [(0.9, 0.3), (0.0, 0.1), (0.5, 0.2), (-0.7, 0.4)])
@pytest.mark.parametrize("offset", [0.5, 4.0, 8.9, 9.05, 15.0, 40.0])
def test_grid_phi_roundtrip_spans_both_regions(a, e, offset):
    """phi_inv o phi is the identity on both sides of the p_sep + 9 seam."""
    pytest.importorskip("few")
    x = _block(a, _separatrix(a, e) + offset, e)
    assert np.allclose(GridReparam.phi_inv(GridReparam.phi(x)), x, rtol=1e-8)


@pytest.mark.parametrize("a,e", [(0.9, 0.3), (0.0, 0.1), (-0.7, 0.4)])
def test_grid_phi_is_continuous_across_the_region_seam(a, e):
    """All four grid coords vary smoothly through the region A/B boundary."""
    pytest.importorskip("few")
    bnd = _separatrix(a, e) + 9.001
    lo = GridReparam.phi(_block(a, bnd - 1e-7, e))
    hi = GridReparam.phi(_block(a, bnd + 1e-7, e))
    assert np.allclose(lo, hi, atol=1e-6), f"discontinuity at the seam: {hi - lo}"


@pytest.mark.parametrize("a,e", [(0.9, 0.3), (0.0, 0.1), (-0.7, 0.4)])
def test_grid_phi_slope_is_continuous_across_the_region_seam(a, e):
    """du/dp matches either side, so log_abs_det_jac has no step at the seam."""
    pytest.importorskip("few")
    bnd = _separatrix(a, e) + 9.001
    h = 1e-5

    def slope(p0):
        return (GridReparam.phi(_block(a, p0 + h, e))[3]
                - GridReparam.phi(_block(a, p0 - h, e))[3]) / (2 * h)

    below, above = slope(bnd - 1e-3), slope(bnd + 1e-3)
    assert np.isclose(above / below, 1.0, rtol=1e-3)


@pytest.mark.parametrize("p", [8.0, 12.0, 30.0])
def test_grid_reparam_to_x_inverts_to_u(p):
    """The sampler's u -> physical map recovers the truth it was built at."""
    pytest.importorskip("few")
    truth = _vector(0.9, p, 0.3)
    rp = GridReparam.from_covariance(12, np.arange(6), GRID_COV, truth[:6])
    assert np.allclose(rp.to_x(rp.to_u(truth)), truth, rtol=1e-8)


def test_grid_phi_is_monotonic_in_p_through_the_seam():
    pytest.importorskip("few")
    a, e = 0.9, 0.3
    ps = np.linspace(_separatrix(a, e) + 0.01, 150.0, 2000)
    u = np.array([GridReparam.phi(_block(a, p, e))[3] for p in ps])
    assert np.all(np.diff(u) > 0)


@pytest.mark.parametrize("p_of_sep", [-0.5, 300.0])
def test_grid_phi_inv_returns_nan_outside_the_grid_domain(p_of_sep):
    """Proposals below the separatrix or past FEW's p=200 grid edge are rejected."""
    pytest.importorskip("few")
    a, e = 0.9, 0.3
    p = _separatrix(a, e) + p_of_sep if p_of_sep < 0 else p_of_sep
    v = GridReparam.phi(_block(a, p, e))
    assert np.any(np.isnan(GridReparam.phi_inv(v)))


@pytest.mark.parametrize("p_of_sep", [-0.5, 300.0])
def test_grid_reparam_rejects_an_out_of_domain_injection(p_of_sep):
    pytest.importorskip("few")
    a, e = 0.9, 0.3
    p = _separatrix(a, e) + p_of_sep if p_of_sep < 0 else p_of_sep
    truth = _vector(a, p, e)
    with pytest.raises(ValueError, match="grid"):
        GridReparam.from_covariance(12, np.arange(6), GRID_COV, truth[:6])


def test_grid_log_abs_det_jac_is_finite_in_both_regions():
    pytest.importorskip("few")
    for p in (8.0, 30.0):
        truth = _vector(0.9, p, 0.3)
        rp = GridReparam.from_covariance(12, np.arange(6), GRID_COV, truth[:6])
        assert np.isfinite(rp.log_abs_det_jac(rp.to_u(truth)))


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(11)
    A = rng.standard_normal((3, 3))
    cov = A @ A.T + 0.1 * np.eye(3)
    rp = Reparam.from_covariance(6, [0, 1, 2], cov, np.array([1.0, 2.0, 3.0]))
    path = str(tmp_path / "reparam.npz")
    rp.save(path, "auto")
    rp2, mode = Reparam.load(path)
    assert mode == "auto"
    assert np.allclose(rp2.R, rp.R)
    assert np.allclose(rp2.sig, rp.sig)
    assert np.allclose(rp2.mu, rp.mu)
