"""Executa UMA simulação em processo próprio (o NEURON é um singleton por processo).

Uso:
  python -m neurochaos.worker --config configs/mvp.yaml --out results/teste \
      --set coupling.K=1.5 --set connectome.topology=shuffled
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import yaml

from .config import load_config


def parse_sets(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--overrides-json", default=None, help="arquivo JSON com overrides pontuados")
    ap.add_argument("--no-arrays", action="store_true")
    a = ap.parse_args(argv)
    ov = {}
    if a.overrides_json:
        ov.update(json.loads(Path(a.overrides_json).read_text(encoding="utf-8")))
    ov.update(parse_sets(a.set))
    cfg = load_config(a.config, ov)
    from .experiment import run_job  # importa NEURON só aqui

    try:
        run_job(cfg, a.out, save_arrays=not a.no_arrays)
    except Exception:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())