from dataclasses import replace

import numpy as np
import pytest

from analytic_benchmark import benchmark, periodic_rollout, yee_reference
from inference import ParameterVector, fit_one_parameter, relative_l2
from maxwell_fdtd import ArenaConfig, arena_grid


def test_continuum_reference_and_second_order_convergence():
    coarse, fine = benchmark(12), benchmark(24)
    for name in ("electric_relative_l2", "magnetic_relative_l2"):
        # Second-order Yee should reduce error by ~4; [3,5] allows pre-asymptotic terms.
        assert 3 < coarse[name] / fine[name] < 5
        assert fine[name] < 0.01  # 24 cells/wavelength resolves these smooth modes.
    assert fine["div_e_dimensionless"] < 1e-10
    assert fine["div_h_dimensionless"] < 1e-10
    assert fine["relative_energy_drift"] < 1e-4


def test_parameter_recovery_from_independent_analytic_fields():
    n = 24
    _, _, times, dt = periodic_rollout(n)
    e, h = yee_reference(n, times[0], dt)
    observations = (e[None, :-1, :-1, :-1], h[None, :-1, :-1, :-1])
    params = ParameterVector(("epsilon_r",), (3.0,), ("1",), ((2.5, 5.5),))
    fit = fit_one_parameter(lambda eps: periodic_rollout(n, epsilon=eps)[:2], observations, params)
    # Continuum vs discrete phase velocity causes O(dx²) bias, not exact recovery.
    assert abs(fit["value"] - 4) / 4 < 0.02


def test_configuration_and_physical_bounds():
    for kwargs in [
        dict(frames=0),
        dict(stride=-1),
        dict(skip=-1),
        dict(length=0),
        dict(seed=-1),
    ]:
        with pytest.raises(ValueError):
            ArenaConfig(**kwargs)
    with pytest.raises(ValueError):
        arena_grid(-1)
    with pytest.raises(ValueError):
        arena_grid(float("nan"))
    with pytest.raises(ValueError):
        arena_grid(4, config=replace(ArenaConfig(), full_size=32))
    c = ArenaConfig()
    np.testing.assert_allclose(c.magnetic_times - c.times, c.dt / 2, rtol=1e-12)
    assert c.coordinates().shape == (32, 32, 32, 3, 3)
    np.testing.assert_allclose(c.coordinates()[0, 0, 0, 0], np.array([16.5, 16, 16]) * c.dx)


def test_errors_and_parameter_vector_validation():
    with pytest.raises(ValueError):
        ParameterVector(("eps",), (0.0,), ("1",), ((1.0, 10.0),))
    with pytest.raises(ValueError):
        ParameterVector(("eps",), (1.0, 2.0), ("1",), ((1.0, 10.0),))
    with pytest.raises(ValueError):
        relative_l2(np.ones(3), np.ones(4))
    assert relative_l2(np.zeros(3), np.zeros(3)) == 0
    assert np.isinf(relative_l2(np.ones(3), np.zeros(3)))
