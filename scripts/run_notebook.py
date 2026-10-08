"""Execute the full notebook in a fresh kernel using this Python interpreter."""

import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lab1-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "lab1-cache"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(Path(tempfile.gettempdir()) / "lab1-jupyter"))

import nbformat  # noqa: E402
from jupyter_client import KernelManager  # noqa: E402
from nbclient import NotebookClient  # noqa: E402


def main():
    path = ROOT / "lab1_maxwell.ipynb"
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    client = NotebookClient(
        notebook, km=manager, timeout=300, resources={"metadata": {"path": str(ROOT)}}
    )
    started = time.perf_counter()
    client.execute()
    nbformat.write(notebook, path)
    report = {
        "success": True,
        "python": platform.python_version(),
        "seconds": time.perf_counter() - started,
        "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
        "all_executed": all(
            c.execution_count is not None for c in notebook.cells if c.cell_type == "code"
        ),
    }
    (ROOT / "results/notebook_execution.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
