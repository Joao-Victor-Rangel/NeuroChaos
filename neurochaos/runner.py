"""Execução paralela de jobs, um subprocesso por simulação (NEURON não é reentrante)."""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import ROOT


def run_one(config, out_dir, overrides, no_arrays=False, timeout=None):
    out_dir = Path(out_dir)
    if (out_dir / "metrics.json").exists():
        return out_dir, "cached", 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    ov_file = out_dir / "overrides.json"
    ov_file.write_text(json.dumps(overrides, indent=1), encoding="utf-8")
    cmd = [sys.executable, "-m", "neurochaos.worker", "--config", str(config), "--out", str(out_dir),
           "--overrides-json", str(ov_file)]
    if no_arrays:
        cmd.append("--no-arrays")
    t0 = time.time()
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    with open(out_dir / "log.txt", "w", encoding="utf-8") as log:
        p = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, env=env)
    status = "ok" if p.returncode == 0 and (out_dir / "metrics.json").exists() else "error"
    return out_dir, status, time.time() - t0


def run_jobs(config, jobs, n_workers=None, no_arrays=False):
    """jobs: lista de (out_dir, overrides). Retorna lista de (out_dir, status, segundos)."""
    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(run_one, config, d, ov, no_arrays): d for d, ov in jobs}
        for k, f in enumerate(as_completed(futs), 1):
            d, st, dt = f.result()
            results.append((d, st, dt))
            print(f"[{k}/{len(jobs)}] {st:6s} {dt:6.1f}s  {Path(d).name}", flush=True)
    return results