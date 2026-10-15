"""Metadados de reprodutibilidade: versões, hash do código e configuração (guia §8.1, §9)."""

import hashlib
import json
import platform
import subprocess

from .config import ROOT


def git_hash():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        h = out.stdout.strip() or "unknown"
        return h + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def config_hash(cfg):
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def versions():
    v = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("neuron", "netpyne", "numpy", "scipy"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = None
    return v


def provenance(cfg):
    return {"git": git_hash(), "config_hash": config_hash(cfg), "versions": versions()}
