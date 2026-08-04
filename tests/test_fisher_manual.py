from types import SimpleNamespace

import numpy as np
import pytest

from emridispatch.fisher import get_fisher_provider
from emridispatch.parameters import INTRINSIC_ORDER

SIGMAS = np.array([2e-3, 1e-5, 1e-6, 1e-5, 1e-6, 4e4])


def _cfg(path):
    return SimpleNamespace(prior=SimpleNamespace(
        fisher="manual", sigmas=None, covariance_file=str(path)))


def test_a_covariance_file_without_an_order_key_is_rejected(tmp_path):
    path = tmp_path / "cov.npz"
    np.savez(path, cov=np.diag(SIGMAS ** 2))

    with pytest.raises(ValueError, match="order"):
        get_fisher_provider(_cfg(path))


def test_a_covariance_file_in_intrinsic_order_is_accepted(tmp_path):
    path = tmp_path / "cov.npz"
    np.savez(path, cov=np.diag(SIGMAS ** 2), order=np.array(INTRINSIC_ORDER))

    provider = get_fisher_provider(_cfg(path))
    res = provider.compute({}, duration=1.0, delta_t=10.0)
    np.testing.assert_allclose([res.sigmas[p] for p in INTRINSIC_ORDER], SIGMAS)
