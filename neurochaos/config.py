"""Carregamento de configuração YAML com calibração e overrides por chave pontuada."""

import copy
import json
import warnings
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def deep_update(base, upd):
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg, key, value):
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def get_dotted(cfg, key, default=None):
    node = cfg
    for p in key.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(path, overrides=None, use_calibration=True):
    """Lê o YAML base, aplica o arquivo de calibração (se existir) e depois os overrides.

    `overrides` é um dict {"coupling.K": 1.5, ...}; tem precedência sobre a calibração.
    """
    with open(resolve(path), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg.get("inherit"):  # herda de outro YAML e sobrescreve só o que for declarado aqui
        parent = load_config(cfg["inherit"], use_calibration=False)
        parent.pop("calibration_file", None)
        child = {k: v for k, v in cfg.items() if k != "inherit"}
        cfg = deep_update(parent, child)
    cal = cfg.get("calibration_file")
    if use_calibration and cal:
        p = resolve(cal)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                cfg = deep_update(cfg, json.load(f)["config_updates"])
            cfg["_calibration_applied"] = str(cal)
        else:
            warnings.warn(f"Arquivo de calibração {cal} não encontrado; usando valores do YAML.")
            cfg["_calibration_applied"] = None
    for k, v in (overrides or {}).items():
        set_dotted(cfg, k, v)
    return cfg
