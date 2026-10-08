"""Import ordinary project modules without installing a package."""

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lab1-mpl"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
