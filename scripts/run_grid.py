"""Executa uma grade (retomável: jobs com metrics.json existente são pulados).

  python scripts/run_grid.py --grid configs/grid_debug.yaml --workers 3
  python scripts/run_grid.py --grid configs/grid_debug.yaml --dry-run
"""

import argparse
import csv
import json

import _bootstrap  # noqa: F401

from neurochaos.config import resolve
from neurochaos.grid import expand, load_grid
from neurochaos.runner import run_jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="executa só os N primeiros jobs")
    ap.add_argument("--family", choices=["main", "controls"], default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-arrays", action="store_true", help="não salvar data.npz (economiza disco)")
    a = ap.parse_args()

    grid = load_grid(a.grid)
    root = resolve("results") / grid["name"]
    jobs = expand(grid, root)
    if a.family:
        jobs = [j for j in jobs if j[3] == a.family]
    if a.limit:
        jobs = jobs[: a.limit]
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "jobs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["family", "out_dir", "condition"])
        for d, ov, cond, fam in jobs:
            w.writerow([fam, d.relative_to(root), json.dumps(cond)])
    (root / "grid.json").write_text(json.dumps(grid, indent=1), encoding="utf-8")
    print(f"{len(jobs)} jobs em {root}")
    if a.dry_run:
        return
    res = run_jobs(grid["base_config"], [(d, ov) for d, ov, _, _ in jobs], n_workers=a.workers,
                   no_arrays=a.no_arrays)
    bad = [str(d) for d, st, _ in res if st == "error"]
    print(f"concluído: {len(res) - len(bad)} ok, {len(bad)} com erro")
    for b in bad[:20]:
        print("  erro:", b)


if __name__ == "__main__":
    main()