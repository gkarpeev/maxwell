"""SymPy/SymEngine comparison on the same scalar part of Maxwell's curl."""

import time

import numpy as np
import symengine as se
import sympy as sp


def compare_libraries(repeats: int = 100) -> dict:
    """Compare differentiation, expansion, substitution, not whole vector APIs.

    Use an explicit 3D polynomial h(x,y,z,t); this avoids measuring differences
    in unevaluated-function support. Startup/conversion is outside the timing.
    """
    x, y, z, t, c, eps = sp.symbols("x y z t c epsilon_r")
    hy, hz = x * z**2 * t + y**2, x * y * t + z**2
    sympy_input = c / eps * (sp.diff(hz, y) - sp.diff(hy, z))
    symengine_input = se.sympify(sympy_input)
    started = time.perf_counter()
    for _ in range(repeats):
        py_result = sp.expand(sp.diff(sympy_input, t))
    py_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        engine_result = se.expand(se.diff(symengine_input, se.Symbol("t")))
    engine_seconds = time.perf_counter() - started
    equivalent = sp.simplify(py_result - sp.sympify(engine_result)) == 0
    substitutions = dict(zip((x, y, z, t, c, eps), (0.2, 0.3, 0.4, 0.1, 1.0, 4.0)))
    py_value = float(py_result.subs(substitutions))
    engine_value = float(engine_result.subs({se.sympify(k): v for k, v in substitutions.items()}))
    return {
        "expression": str(sympy_input),
        "derivative": str(py_result),
        "equivalent": bool(equivalent),
        "numeric_difference": abs(py_value - engine_value),
        "sympy_seconds": py_seconds,
        "symengine_seconds": engine_seconds,
        "repeats": repeats,
        "versions": {"sympy": sp.__version__, "symengine": se.__version__},
        "finite": bool(np.isfinite(py_value) and np.isfinite(engine_value)),
    }
