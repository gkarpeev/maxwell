"""Parameter vector and optimization, independent of the physical solver."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class ParameterVector:
    names: tuple[str, ...]
    values: tuple[float, ...]
    units: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]

    def __post_init__(self):
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("Parameter names must be nonempty and unique")
        if len({len(self.names), len(self.values), len(self.units), len(self.bounds)}) != 1:
            raise ValueError("Parameter metadata lengths disagree")
        for value, (lower, upper) in zip(self.values, self.bounds):
            if (
                not all(np.isfinite(v) for v in (value, lower, upper))
                or not lower < upper
                or not lower <= value <= upper
            ):
                raise ValueError("Finite parameters must lie within ordered bounds")

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values))


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    if prediction.shape != reference.shape:
        raise ValueError("Prediction and reference shapes differ")
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(reference)):
        raise ValueError("Nonfinite field")
    numerator = np.linalg.norm(prediction - reference)
    denominator = np.linalg.norm(reference)
    if denominator == 0:
        return 0.0 if numerator == 0 else float("inf")
    return float(numerator / denominator)


def fit_one_parameter(
    forward: Callable[[float], tuple[np.ndarray, ...]],
    observations: tuple[np.ndarray, ...],
    parameters: ParameterVector,
    tolerance: float = 1e-4,
) -> dict:
    """Bounded scalar search; observations contain only the training split.

    Each field contributes its own normalized squared error with equal weight.
    This routine intentionally does not promise multivariate/global optimization.
    """
    if len(parameters.names) != 1 or not observations:
        raise ValueError("This optimizer expects one parameter and nonempty observations")
    if any(np.linalg.norm(field) == 0 for field in observations):
        raise ValueError("Cannot normalize an identically zero observed field")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance must be positive")
    history = []

    def objective(value):
        predictions = forward(float(value))
        if len(predictions) != len(observations):
            raise ValueError("Forward model returned the wrong number of fields")
        loss = float(np.mean([relative_l2(p, y) ** 2 for p, y in zip(predictions, observations)]))
        history.append([float(value), loss])
        return loss

    result = minimize_scalar(
        objective,
        bounds=parameters.bounds[0],
        method="bounded",
        options={"xatol": tolerance},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Optimization failed: {result.message}")
    return {
        "name": parameters.names[0],
        "value": float(result.x),
        "loss": float(result.fun),
        "evaluations": int(result.nfev),
        "history": history,
    }
