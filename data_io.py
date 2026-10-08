"""Pinned HTTP downloads, hashes and data adapters; no import-time I/O."""

import hashlib
import json
import urllib.request
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_data(directory: Path | None = None) -> dict[str, Path]:
    """Download ~412 MB once, atomically publish only files matching the manifest.

    Existing corrupt files cause an explicit error; they are never silently
    accepted or overwritten. Remove a corrupt cache entry before retrying.
    """
    directory = ROOT / "data/raw" if directory is None else Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "sources/data_manifest.json").read_text())
    paths = {}
    for name, entry in manifest.items():
        path = directory / name
        if path.exists():
            if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
                raise ValueError(f"Corrupt cached data: {path}")
        else:
            temporary = path.with_suffix(path.suffix + ".part")
            try:
                request = urllib.request.Request(
                    entry["url"], headers={"User-Agent": "maxwell-lab1/1"}
                )
                with (
                    urllib.request.urlopen(request, timeout=120) as response,
                    temporary.open("wb") as stream,
                ):
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(block)
                if (
                    temporary.stat().st_size != entry["bytes"]
                    or sha256(temporary) != entry["sha256"]
                ):
                    raise ValueError(f"Downloaded data failed checksum: {name}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        paths[name] = path
    return paths


def load_arena(path: Path, samples: tuple[int, ...] = (0, 1)) -> tuple[np.ndarray, np.ndarray]:
    if not samples or any(type(s) is not int or s < 0 for s in samples):
        raise ValueError("Sample indices must be nonnegative integers")
    with h5py.File(path) as h5:
        electric, magnetic = h5["test/d_field"], h5["test/h_field"]
        if electric.shape != magnetic.shape or electric.shape[1:] != (8, 32, 32, 32, 3):
            raise ValueError("Unexpected PDEArena shape")
        if max(samples) >= electric.shape[0]:
            raise ValueError("Sample index out of range")
        e = np.stack([electric[s] for s in samples])
        h = np.stack([magnetic[s] for s in samples])
    if not np.all(np.isfinite(e)) or not np.all(np.isfinite(h)):
        raise ValueError("Nonfinite external observations")
    return e, h


if __name__ == "__main__":
    for name, path in ensure_data().items():
        print(name, path.stat().st_size, "bytes; SHA-256 OK")
