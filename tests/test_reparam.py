import numpy as np

from emridispatch.reparam import Reparam, ReparamCallable


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
