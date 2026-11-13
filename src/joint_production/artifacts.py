"""Content-addressed provenance for private reproducible research runs."""
from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tarfile
from datetime import datetime, timezone


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path: str | Path, value, *, exclusive: bool = True) -> None:
    path = Path(path)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as handle:
        handle.write(encoded)


def source_files(root: Path) -> list[Path]:
    files = []
    for directory in ("src", "scripts", "tests", "configs"):
        files += [p for p in (root / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py", ".toml", ".sh"}]
    for name in ("pyproject.toml", "Makefile", "README.md", "data/manifest.json", "data/README.md"):
        path = root / name
        if path.is_file():
            files.append(path)
    return sorted(set(files))


def capture_run(root: Path, output: Path, *, config: Path, data: Path, split: Path, seeds: list[int], command: list[str] | None = None) -> dict:
    """Claim a fresh run directory and preserve the actual source (even uncommitted)."""
    root, output = root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    files = source_files(root)
    source_bytes = {str(p.relative_to(root)): p.read_bytes() for p in files}
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in source_bytes.items()}
    config_bytes = config.read_bytes()
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "torch", "pandas", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    manifest = {
        "schema_version": 1, "run_id": output.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "--verify", "HEAD"),
        "git_status": git("status", "--porcelain=v1"),
        "source_hashes": hashes, "source_sha256": canonical_hash(hashes),
        "config_path": str(config), "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "data_sha256": sha256_file(data), "split_sha256": sha256_file(split),
        "command": command or [sys.executable, *sys.argv], "seeds": seeds,
        "host": socket.gethostname(), "platform": platform.platform(),
        "python": sys.version, "python_executable": sys.executable,
        "packages": versions,
        "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "test_used_for_selection": False,
    }
    manifest["git_dirty"] = bool(manifest["git_status"]) or manifest["git_commit"] is None
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for name, content in source_bytes.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(content))
    (output / 'config.toml').write_bytes(config_bytes)
    manifest["source_archive_sha256"] = sha256_file(output / "source.tar.gz")
    write_json(output / "manifest.json", manifest)
    return manifest
