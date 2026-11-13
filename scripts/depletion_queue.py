"""Run a list of depletion_dev experiments with memory guardrails.

    python scripts/depletion_queue.py configs/<queue>.toml --outputs outputs

The TOML has ``lanes`` (max concurrent jobs, default 2), ``min_available_gb``
(default 40) and a list of [[job]] tables with ``label`` and ``args`` (the
depletion_dev.py argument list without --label/--output). Each finished run is
scored immediately. Every process is capped by JPF_GPU_MEMORY_FRACTION.
A job never starts while available memory is below the threshold.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 ** 2
    return float("inf")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue")
    parser.add_argument("--outputs", default="outputs")
    args = parser.parse_args(argv)
    spec = tomllib.loads(Path(args.queue).read_text())
    lanes = int(spec.get("lanes", 2))
    threshold = float(spec.get("min_available_gb", 40))
    outputs = Path(args.outputs)
    pending = list(spec["job"])
    running: list[tuple[str, subprocess.Popen]] = []
    env = dict(os.environ, JPF_GPU_MEMORY_FRACTION=os.environ.get("JPF_GPU_MEMORY_FRACTION", "0.25"))
    while pending or running:
        for label, proc in list(running):
            if proc.poll() is not None:
                running.remove((label, proc))
                print(f"[queue] finished {label} rc={proc.returncode}", flush=True)
        if pending and len(running) < lanes and available_gb() >= threshold:
            job = pending.pop(0)
            label = job["label"]
            out = outputs / job.get("output", label)
            run = [sys.executable, "scripts/depletion_dev.py", *shlex.split(job["args"]), "--label", label,
                   "--output", str(out)]
            score = [sys.executable, "scripts/depletion_dev.py", "score", "--runs", str(out),
                     "--output", str(outputs / f"score_{job.get('output', label)}")]
            command = " ".join(shlex.quote(c) for c in run) + " && " + " ".join(shlex.quote(c) for c in score)
            outputs.mkdir(parents=True, exist_ok=True)
            log = open(outputs / f"queue_{job.get('output', label)}.log", "w")
            running.append((label, subprocess.Popen(["bash", "-c", command], cwd=ROOT, env=env,
                                                    stdout=log, stderr=subprocess.STDOUT)))
            print(f"[queue] started {label} available={available_gb():.0f}G", flush=True)
            time.sleep(30)  # let the new job allocate before the next memory check
            continue
        time.sleep(15)
    print("[queue] all done", flush=True)


if __name__ == "__main__":
    main()
