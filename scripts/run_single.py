"""Uma execução com overrides e figura-resumo.

  python scripts/run_single.py --out results/single/theta004 --set stimulus.theta.amp_nA=0.04
"""

import argparse
import subprocess
import sys

import _bootstrap  # noqa: F401

from neurochaos.config import ROOT, resolve
from neurochaos.runner import run_one
from neurochaos.worker import parse_sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mvp.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    out, st, sec = run_one(a.config, resolve(a.out), parse_sets(a.set))
    print(f"{st} em {sec:.1f}s -> {out}")
    if st != "error":
        subprocess.run([sys.executable, str(ROOT / "scripts" / "plot_run.py"), str(out)], check=False)


if __name__ == "__main__":
    main()