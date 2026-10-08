"""Independent continuum solution used to verify the fdtd library solver.

Three transverse modes vary in all spatial directions. Values are evaluated
analytically at each Yee component's own location and time. No numerical
solver creates the reference data.
"""

import fdtd
import numpy as np

from inference import relative_l2

C = 299792458.0
E_OFFSETS = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
H_OFFSETS = np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
MODES = [
    (np.array(k, dtype=float), np.array(a, dtype=float), amplitude)
    for k, a, amplitude in [
        ((1, 1, 0), (1, -1, 0), 1.0),
        ((0, 1, 1), (0, 1, -1), 0.7),
        ((1, 0, 1), (1, 0, -1), 0.4),
    ]
]


def plane_waves(points: np.ndarray, time: float, epsilon: float = 4.0, length: float = 0.03):
    """Continuum e,h at collocated points in normalized fdtd field units."""
    electric = np.zeros(points.shape, dtype=float)
    magnetic = np.zeros(points.shape, dtype=float)
    for mode, polarization, amplitude in MODES:
        k = 2 * np.pi / length * mode
        omega = C * np.linalg.norm(k) / np.sqrt(epsilon)
        a = amplitude * polarization / np.linalg.norm(polarization)
        b = C / omega * np.cross(k, a)
        phase = np.cos(points @ k - omega * time)[..., None]
        electric += phase * a
        magnetic += phase * b
    return electric, magnetic


def yee_reference(n: int, time: float, dt: float, epsilon: float = 4.0, length: float = 0.03):
    indices = np.stack(np.meshgrid(*([np.arange(n + 1)] * 3), indexing="ij"), axis=-1)
    electric = np.empty((n + 1, n + 1, n + 1, 3))
    magnetic = np.empty_like(electric)
    for j in range(3):
        electric[..., j] = plane_waves(
            (indices + E_OFFSETS[j]) * length / n, time, epsilon, length
        )[0][..., j]
        magnetic[..., j] = plane_waves(
            (indices + H_OFFSETS[j]) * length / n, time + dt / 2, epsilon, length
        )[1][..., j]
    return electric, magnetic


def periodic_rollout(
    n: int,
    epsilon: float = 4.0,
    initial_epsilon: float = 4.0,
    steps: tuple[int, ...] | None = None,
):
    """Source-free 3cm periodic cube. Last index is a periodic ghost plane.

    The initial fields are held fixed by initial_epsilon when epsilon is fitted.
    Courant=0.4 and n timesteps give the same final physical time on every mesh.
    """
    if type(n) is not int or n < 4 or epsilon < 1 or not np.isfinite(epsilon):
        raise ValueError("Need n >= 4 and finite epsilon >= 1")
    steps = (n,) if steps is None else steps
    if (
        not steps
        or any(type(s) is not int or s < 1 for s in steps)
        or tuple(sorted(set(steps))) != steps
    ):
        raise ValueError("Steps must be strictly increasing positive integers")
    length = 0.03
    fdtd.set_backend("numpy")
    grid = fdtd.Grid(
        (n + 1,) * 3, grid_spacing=length / n, permittivity=epsilon, courant_number=0.4
    )
    grid[0, :, :] = fdtd.PeriodicBoundary(name="xbounds")
    grid[:, 0, :] = fdtd.PeriodicBoundary(name="ybounds")
    grid[:, :, 0] = fdtd.PeriodicBoundary(name="zbounds")
    grid.E[:], grid.H[:] = yee_reference(n, 0.0, grid.time_step, initial_epsilon, length)
    electric, magnetic = [], []
    previous = 0
    for step in steps:
        grid.run(step - previous, progress_bar=False)
        electric.append(grid.E[:-1, :-1, :-1].copy())
        magnetic.append(grid.H[:-1, :-1, :-1].copy())
        previous = step
    return (
        np.asarray(electric),
        np.asarray(magnetic),
        np.asarray(steps) * grid.time_step,
        grid.time_step,
    )


def divergence_yee(field: np.ndarray, dx: float, electric: bool = True) -> np.ndarray:
    """Divergence on unique periodic cells, with correct component offsets."""
    result = np.zeros(field.shape[:-1])
    for axis in range(3):
        component = field[..., axis]
        if electric:
            result += (component - np.roll(component, 1, axis=axis)) / dx
        else:
            result += (np.roll(component, -1, axis=axis) - component) / dx
    return result


def benchmark(n: int) -> dict:
    e, h, times, dt = periodic_rollout(n)
    er, hr = yee_reference(n, float(times[-1]), dt)
    ei, hi = yee_reference(n, 0.0, dt)
    crop = (slice(None, -1),) * 3
    energy_initial = np.sum(4 * ei[crop] ** 2 + hi[crop] ** 2)
    energy_final = np.sum(4 * e[-1] ** 2 + h[-1] ** 2)
    return {
        "n": n,
        "dt_s": dt,
        "time_s": float(times[-1]),
        "electric_relative_l2": relative_l2(e[-1], er[crop]),
        "magnetic_relative_l2": relative_l2(h[-1], hr[crop]),
        "relative_energy_drift": float(abs(energy_final / energy_initial - 1)),
        "div_e_dimensionless": float(np.max(np.abs(divergence_yee(e[-1], 0.03 / n))) * 0.03),
        "div_h_dimensionless": float(np.max(np.abs(divergence_yee(h[-1], 0.03 / n, False))) * 0.03),
    }
