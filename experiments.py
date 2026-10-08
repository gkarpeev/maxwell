"""Reproducible experiments shared by the notebook and command-line runner."""

import json
import platform
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from analytic_benchmark import benchmark, periodic_rollout, yee_reference
from data_io import ROOT, ensure_data, load_arena, sha256
from emval_check import derivative_control, diagnose_cloud, load_emval
from inference import ParameterVector, fit_one_parameter, relative_l2
from maxwell_fdtd import ArenaConfig, simulate_arena


def run_arena(electric: np.ndarray, magnetic: np.ndarray) -> tuple[dict, dict]:
    config = ArenaConfig()
    start = time.perf_counter()
    replay = simulate_arena(10.0)
    report = {
        "replay_seconds": time.perf_counter() - start,
        "replay_e_error": relative_l2(replay[0], electric[0]),
        "replay_h_error": relative_l2(replay[1], magnetic[0]),
        "epsilon_reference": 10.0,
        "fit_sample": 0,
        "fit_frames": [0, 1, 2, 3],
        "holdout_frames_sample0": [4, 5, 6, 7],
        "holdout_sample": 1,
    }
    if max(report["replay_e_error"], report["replay_h_error"]) >= 1e-8:
        raise RuntimeError("Generator replay failed; do not proceed to fitting")
    train_config = replace(config, frames=4)

    def forward(eps):
        return simulate_arena(eps, config=train_config)

    observations = (electric[0, :4], magnetic[0, :4])
    parameters = ParameterVector(("epsilon_r",), (8.0,), ("dimensionless",), ((6.0, 14.0),))
    profile = []
    started = time.perf_counter()
    for value in np.linspace(6, 14, 9):
        prediction = forward(float(value))
        loss = np.mean([relative_l2(p, y) ** 2 for p, y in zip(prediction, observations)])
        profile.append([float(value), float(loss)])
    report["profile_seconds"] = time.perf_counter() - started
    report["profile"] = profile
    started = time.perf_counter()
    fit = fit_one_parameter(forward, observations, parameters)
    report["fit_seconds"] = time.perf_counter() - started
    report["fit"] = fit
    report["epsilon_relative_error"] = abs(fit["value"] - 10) / 10
    predictions = {"baseline": simulate_arena(parameters.values[0])}
    for sample in (0, 1):
        prediction = simulate_arena(fit["value"], sample=sample)
        predictions[f"sample{sample}"] = prediction
        report[f"sample{sample}"] = {
            "e_per_frame": [relative_l2(p, y) for p, y in zip(prediction[0], electric[sample])],
            "h_per_frame": [relative_l2(p, y) for p, y in zip(prediction[1], magnetic[sample])],
            "e_holdout": relative_l2(
                prediction[0][4:] if sample == 0 else prediction[0],
                electric[sample, 4:] if sample == 0 else electric[sample],
            ),
            "h_holdout": relative_l2(
                prediction[1][4:] if sample == 0 else prediction[1],
                magnetic[sample, 4:] if sample == 0 else magnetic[sample],
            ),
        }
    report["baseline_e_error"] = relative_l2(predictions["baseline"][0], electric[0])
    report["baseline_h_error"] = relative_l2(predictions["baseline"][1], magnetic[0])
    report["total_seconds"] = time.perf_counter() - start
    return report, predictions


def run_analytic() -> dict:
    started = time.perf_counter()
    convergence = [benchmark(n) for n in (12, 24, 48)]
    n = 24
    _, _, times, dt = periodic_rollout(n)
    er, hr = yee_reference(n, float(times[0]), dt)
    crop = (slice(None, -1),) * 3
    observations = (er[crop][None], hr[crop][None])
    parameters = ParameterVector(("epsilon_r",), (3.0,), ("1",), ((2.5, 5.5),))
    fit = fit_one_parameter(
        lambda eps: periodic_rollout(n, epsilon=eps)[:2], observations, parameters
    )
    # Later exact state was not used in the inverse problem.
    ep, hp, times, dt = periodic_rollout(n, epsilon=fit["value"], steps=(2 * n,))
    er, hr = yee_reference(n, float(times[0]), dt)
    return {
        "convergence": convergence,
        "continuum_fit": fit,
        "epsilon_relative_error": abs(fit["value"] - 4) / 4,
        "later_e_error": relative_l2(ep[0], er[crop]),
        "later_h_error": relative_l2(hp[0], hr[crop]),
        "seconds": time.perf_counter() - started,
    }


def run_emval(path: Path) -> tuple[dict, object]:
    start = time.perf_counter()
    cloud = load_emval(path)
    report = {
        "frequency_hz": cloud.frequency,
        "port": 0,
        "phasor_sign_documented": False,
        "diagnostics": {},
        "smooth_wave_control": {},
    }
    for neighbors in (40, 60, 100):
        report["diagnostics"][str(neighbors)] = diagnose_cloud(cloud, neighbors)
        report["smooth_wave_control"][str(neighbors)] = derivative_control(cloud, neighbors)
    report["seconds"] = time.perf_counter() - start
    return report, cloud


def save_report(report: dict, directory: Path | None = None) -> Path:
    directory = ROOT / "results" if directory is None else directory
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_sha256": {p.name: sha256(p) for p in sorted(ROOT.glob("*.py"))},
        "data_manifest_sha256": sha256(ROOT / "sources/data_manifest.json"),
    }
    path = directory / "experiment.json"
    path.write_text(json.dumps({**report, "provenance": metadata}, indent=2))
    return path


if __name__ == "__main__":
    paths = ensure_data()
    e, h = load_arena(paths["pdearena.h5"])
    arena, _ = run_arena(e, h)
    emval, _ = run_emval(paths["emval.h5"])
    print(save_report({"arena": arena, "analytic": run_analytic(), "emval": emval}))
