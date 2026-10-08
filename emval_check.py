"""Local diagnostics of independent complex 3D EMVal fields.

This does not solve a PCB or train on the benchmark. Derivatives from a sparse
point cloud are estimates; their sensitivity and conditioning are reported.
No assumption about the undocumented phasor sign is silently imposed.
"""

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import h5py
import numpy as np
from scipy.constants import epsilon_0, mu_0
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class CloudField:
    points: np.ndarray
    electric: np.ndarray
    magnetic: np.ndarray
    frequency: float
    design: dict


def load_emval(path: Path, frequency_index: int = 50, port: int = 0) -> CloudField:
    with h5py.File(path) as h5:
        return CloudField(
            h5["probes_m"][:].astype(float),
            h5["E_real"][port, frequency_index].astype(float)
            + 1j * h5["E_imag"][port, frequency_index],
            h5["H_real"][port, frequency_index].astype(float)
            + 1j * h5["H_imag"][port, frequency_index],
            float(h5["frequencies_hz"][frequency_index]),
            json.loads(h5.attrs["design_json"]),
        )


def region_masks(cloud: CloudField) -> dict[str, np.ndarray]:
    """Conservative regions for the selected board (no vias): exclude interfaces."""
    if cloud.design["vias"] or cloud.design["solder_balls"]:
        raise ValueError("This diagnostic region selector supports boards without vias")
    z_ground = cloud.design["layers"][-1]["thickness_m"]
    z_core = z_ground + cloud.design["layers"][1]["thickness_m"]
    z_top = z_core + cloud.design["layers"][0]["thickness_m"]
    x, y, z = cloud.points.T
    sx, sy = cloud.design["size_m"]
    return {
        "air": z > z_top + 1e-4,
        "core": (np.abs(x) < sx / 2 - 7e-4)
        & (np.abs(y) < sy / 2 - 7e-4)
        & (z > z_ground + 2e-5)
        & (z < z_core - 2e-5),
    }


def local_derivatives(
    points: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
    neighbors: int = 60,
    degree: int = 2,
) -> dict[str, np.ndarray]:
    """Weighted local polynomial fit; jacobian[i,component,coordinate].

    Coordinates are scaled per axis to avoid artificial ill-conditioning from
    meter units and the thin dielectric layer. No derivative is taken across
    a material interface: callers must pass one homogeneous region.
    """
    if degree not in (1, 2, 3):
        raise ValueError("degree must be 1, 2, or 3")
    powers = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    powers += [p for p in product(range(degree + 1), repeat=3) if 2 <= sum(p) <= degree]
    if neighbors < len(powers) or neighbors > len(points):
        raise ValueError("neighbors must support the polynomial and fit the cloud")
    distance, indices = cKDTree(points).query(points[targets], k=neighbors)
    gradients, intercepts, conditions, residuals = [], [], [], []
    for target, dist, ids in zip(targets, distance, indices):
        delta = points[ids] - points[target]
        scale = np.sqrt(np.mean(delta**2, axis=0))
        if np.any(scale <= 0):
            raise ValueError("Neighborhood must span all three coordinates")
        q = delta / scale
        matrix = np.stack([np.prod(q ** np.array(p), axis=1) for p in powers], axis=1)
        weights = np.exp(-2 * (dist / max(dist[-1], 1e-30)) ** 2)
        weighted = matrix * np.sqrt(weights[:, None])
        coefficients, _, rank, singular = np.linalg.lstsq(
            weighted, values[ids] * np.sqrt(weights[:, None]), rcond=None
        )
        if rank != len(powers):
            raise ValueError("Rank-deficient 3D polynomial neighborhood")
        gradients.append((coefficients[1:4] / scale[:, None]).T)
        intercepts.append(coefficients[0])
        conditions.append(singular[0] / singular[-1])
        residuals.append(
            np.linalg.norm(matrix @ coefficients - values[ids]) / np.linalg.norm(values[ids])
        )
    return {
        "jacobian": np.array(gradients),
        "value": np.array(intercepts),
        "condition": np.array(conditions),
        "fit_residual": np.array(residuals),
    }


def curl_from_jacobian(jacobian: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            jacobian[..., 2, 1] - jacobian[..., 1, 2],
            jacobian[..., 0, 2] - jacobian[..., 2, 0],
            jacobian[..., 1, 0] - jacobian[..., 0, 1],
        ),
        axis=-1,
    )


def balanced_residual(left: np.ndarray, right: np.ndarray) -> float:
    """||left-right|| / (||left||+||right||); bounded and scale independent."""
    denominator = np.linalg.norm(left) + np.linalg.norm(right)
    if denominator == 0:
        return 0.0
    return float(np.linalg.norm(left - right) / denominator)


def diagnose_cloud(cloud: CloudField, neighbors: int = 60, degree: int = 2) -> dict:
    report = {}
    omega = 2 * np.pi * cloud.frequency
    for region, mask in region_masks(cloud).items():
        points = cloud.points[mask]
        electric, magnetic = cloud.electric[mask], cloud.magnetic[mask]
        # Fixed deterministic subset, independent of measured field values.
        targets = np.linspace(0, len(points) - 1, min(150, len(points)), dtype=int)
        de = local_derivatives(points, electric, targets, neighbors, degree)
        dh = local_derivatives(points, magnetic, targets, neighbors, degree)
        material = cloud.design["layers"][1]["material"]
        eps = material["eps_r"] if region == "core" else 1.0
        loss_tangent = material["loss_tangent"] if region == "core" else 0.0
        sigma = omega * epsilon_0 * eps * loss_tangent
        ce, ch = curl_from_jacobian(de["jacobian"]), curl_from_jacobian(dh["jacobian"])
        r = {
            "points": len(points),
            "epsilon_r": eps,
            "loss_tangent": loss_tangent,
            "effective_sigma_S_m": sigma,
            "median_condition": float(np.median(de["condition"])),
            "max_condition": float(np.max(de["condition"])),
            "median_E_fit_residual": float(np.median(de["fit_residual"])),
            "median_H_fit_residual": float(np.median(dh["fit_residual"])),
        }
        for sign in (-1, 1):
            faraday = -sign * 1j * omega * mu_0 * magnetic[targets]
            ampere = (sigma + sign * 1j * omega * epsilon_0 * eps) * electric[targets]
            air_ampere = sign * 1j * omega * epsilon_0 * electric[targets]
            r[f"exp_{sign:+d}iwt"] = {
                "faraday": balanced_residual(ce, faraday),
                "ampere_material": balanced_residual(ch, ampere),
                "ampere_air_assumption": balanced_residual(ch, air_ampere),
            }
        report[region] = r
    return report


def derivative_control(cloud: CloudField, neighbors: int = 60) -> dict:
    """Known complex plane wave on the actual point geometry, independent of labels.

    This estimates derivative error only for a smooth mode, not for an arbitrary
    near field. Small control error cannot certify accuracy on sharp PCB fields.
    """
    direction = np.ones(3) / np.sqrt(3)
    polarization = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    omega = 2 * np.pi * cloud.frequency
    report = {}
    for region, mask in region_masks(cloud).items():
        points = cloud.points[mask]
        material = cloud.design["layers"][1]["material"]
        eps = material["eps_r"] if region == "core" else 1.0
        tangent = material["loss_tangent"] if region == "core" else 0.0
        wavenumber = omega * np.sqrt(mu_0 * epsilon_0 * eps * (1 - 1j * tangent))
        phase = np.exp(-1j * wavenumber * (points @ direction))[:, None]
        electric = phase * polarization
        magnetic = phase * np.cross(direction, polarization) * wavenumber / (omega * mu_0)
        targets = np.linspace(0, len(points) - 1, min(150, len(points)), dtype=int)
        de = local_derivatives(points, electric, targets, neighbors)
        dh = local_derivatives(points, magnetic, targets, neighbors)
        exact_e = -1j * wavenumber * electric[targets, :, None] * direction
        exact_h = -1j * wavenumber * magnetic[targets, :, None] * direction
        report[region] = {
            "jacobian_e_relative_l2": float(
                np.linalg.norm(de["jacobian"] - exact_e) / np.linalg.norm(exact_e)
            ),
            "jacobian_h_relative_l2": float(
                np.linalg.norm(dh["jacobian"] - exact_h) / np.linalg.norm(exact_h)
            ),
            "faraday": balanced_residual(
                curl_from_jacobian(de["jacobian"]),
                -1j * omega * mu_0 * magnetic[targets],
            ),
            "ampere": balanced_residual(
                curl_from_jacobian(dh["jacobian"]),
                1j * omega * epsilon_0 * eps * (1 - 1j * tangent) * electric[targets],
            ),
        }
    return report


if __name__ == "__main__":
    cloud = load_emval(Path("data/raw/emval.h5"))
    report = {"frequency_hz": cloud.frequency, "phasor_sign_documented": False}
    for k in (40, 60, 100):
        report[str(k)] = diagnose_cloud(cloud, k)
        report[f"control_{k}"] = derivative_control(cloud, k)
    Path("results/emval_probe.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
