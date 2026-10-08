import numpy as np
import pytest

from emval_check import curl_from_jacobian, local_derivatives


def test_quadratic_derivatives_on_irregular_3d_cloud():
    points = np.random.default_rng(123).uniform(-1, 1, (300, 3)) * np.array([0.004, 0.004, 0.0001])
    x, y, z = points.T
    values = np.stack([y * z, x * z, x * y], axis=-1).astype(complex) * (1 + 2j)
    targets = np.arange(0, 300, 20)
    actual = local_derivatives(points, values, targets, neighbors=50)
    x, y, z = points[targets].T
    expected = np.zeros((len(targets), 3, 3), dtype=complex)
    expected[:, 0, 1], expected[:, 0, 2] = z, y
    expected[:, 1, 0], expected[:, 1, 2] = z, x
    expected[:, 2, 0], expected[:, 2, 1] = y, x
    expected *= 1 + 2j
    np.testing.assert_allclose(actual["jacobian"], expected, rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(curl_from_jacobian(actual["jacobian"]), 0, atol=1e-13)


def test_planar_points_rejected():
    points = np.random.default_rng(12).normal(size=(100, 3))
    points[:, 2] = 0
    with pytest.raises(ValueError, match="three coordinates"):
        local_derivatives(points, np.ones((100, 3)), np.arange(5))
