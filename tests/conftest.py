"""Run source-tree checks without installing a mutable development package."""
import os
import sys
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
