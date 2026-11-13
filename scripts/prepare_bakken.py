"""Curate Bakken and accept its immutable grouped split without installation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from joint_production.data import main

if __name__ == "__main__":
    main()
