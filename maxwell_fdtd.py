"""Small adapter for the published PDEArena Maxwell generator (MIT).

Source: sources/pdearena-maxwell.py; source placement and RNG order are kept
identical. The saved d_field is fdtd's normalized E, not electric displacement.
"""

from dataclasses import dataclass

import fdtd
import numpy as np


@dataclass(frozen=True)
class ArenaConfig:
    """Published configuration; lengths in m, time in s, relative ε and μ."""

    seed: int = 126488
    full_size: int = 64
    crop_size: int = 32
    length: float = 3.2e-5
    wavelength: float = 1e-5
    skip: int = 250
    stride: int = 25
    frames: int = 8

    def __post_init__(self):
        for name in ("full_size", "crop_size", "stride", "frames"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.skip) is not int or self.skip < 0:
            raise ValueError("skip must be a nonnegative integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a valid RandomState integer seed")
        if any(not np.isfinite(v) or v <= 0 for v in (self.length, self.wavelength)):
            raise ValueError("length and wavelength must be finite and positive")

    @property
    def c(self) -> float:
        return 299792458.0

    @property
    def dx(self) -> float:
        return self.length / self.full_size

    @property
    def dt(self) -> float:
        return 0.99 / np.sqrt(3) * self.dx / self.c

    @property
    def times(self) -> np.ndarray:
        """Electric-field times; H is staggered forward by half a timestep."""
        return (self.skip + self.stride * np.arange(1, self.frames + 1)) * self.dt

    @property
    def magnetic_times(self) -> np.ndarray:
        return self.times + self.dt / 2

    def coordinates(self, electric: bool = True) -> np.ndarray:
        """Coordinates in m, shape (x,y,z,component,coordinate)."""
        offsets = np.eye(3) / 2 if electric else (1 - np.eye(3)) / 2
        base = np.arange(self.crop_size) + (self.full_size - self.crop_size) // 2
        indices = np.stack(np.meshgrid(base, base, base, indexing="ij"), axis=-1)
        return (indices[..., None, :] + offsets) * self.dx


DEFAULT_CONFIG = ArenaConfig()


def arena_grid(epsilon: float, sample: int = 0, config: ArenaConfig = DEFAULT_CONFIG):
    """Reconstruct full periodic domain, zero IC and 18 known plane sources.

    E and H use the fdtd Yee convention; never impose periodic BC on the crop.
    The seed-to-source mapping must be checked against the external HDF5 file.
    """
    if not np.isfinite(epsilon) or epsilon < 1:
        raise ValueError("epsilon_r must be finite and >= 1 for the fixed CFL step")
    if not isinstance(sample, int) or sample < 0:
        raise ValueError("sample must be a nonnegative integer")
    if (config.full_size, config.crop_size) != (64, 32):
        raise ValueError("Published source placement requires full=64 and crop=32")
    fdtd.set_backend("numpy")
    seed = np.random.RandomState(config.seed).randint(np.iinfo(np.int32).max, size=sample + 1)[-1]
    rng = np.random.RandomState(sample + seed)
    grid = fdtd.Grid(
        (config.length,) * 3,
        grid_spacing=config.dx,
        permittivity=epsilon,
        permeability=1.0,
    )
    grid[0, :, :] = fdtd.PeriodicBoundary(name="xbounds")
    grid[:, 0, :] = fdtd.PeriodicBoundary(name="ybounds")
    grid[:, :, 0] = fdtd.PeriodicBoundary(name="zbounds")
    for normal, axes in [(2, "xy"), (1, "xz"), (0, "yz")]:
        for i in range(6):
            index = []
            for axis in range(3):
                if axis == normal:
                    index.append(int(rng.randint(0, 16)))
                else:
                    length = int(rng.randint(2, 6))
                    start = int(rng.randint(0, 16 - length))
                    index.append(slice(start, start + length))
            amplitude = rng.rand()
            phase = rng.uniform(0, 2 * np.pi)
            polarization = axes[rng.randint(0, 2)]
            period = config.wavelength / config.c * rng.uniform(0.001, 1e3)
            grid[tuple(index)] = fdtd.PlaneSource(
                period=period,
                amplitude=amplitude,
                name=f"planesource{axes}{i}",
                phase_shift=phase,
                polarization=polarization,
            )
    return grid


def simulate_arena(
    epsilon: float, sample: int = 0, config: ArenaConfig = DEFAULT_CONFIG
) -> tuple[np.ndarray, np.ndarray]:
    """Return E/H arrays (time,x,y,z,component), without modifying observations."""
    grid = arena_grid(epsilon, sample, config)
    offset = (config.full_size - config.crop_size) // 2
    crop = (slice(offset, -offset),) * 3
    electric, magnetic = [], []
    grid.run(config.skip, progress_bar=False)
    for _ in range(config.frames):
        grid.run(config.stride, progress_bar=False)
        electric.append(grid.E[crop].copy())
        magnetic.append(grid.H[crop].copy())
    return np.asarray(electric), np.asarray(magnetic)
