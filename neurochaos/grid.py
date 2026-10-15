"""Expansão de grades experimentais (produto cartesiano de eixos × sementes)."""

from itertools import product

import yaml

from .config import resolve

ABBR = {"connectome.topology": "topo", "coupling.K": "K", "stimulus.theta.amp_nA": "th",
        "stimulus.gamma.amp_nA": "ga", "geometry.scale": "geo", "connectome.velocity_m_per_s": "v",
        "connectome.ell_mm": "ell", "connectome.p0": "p0"}


def _slug(cond):
    parts = []
    for k, v in cond.items():
        parts.append(f"{ABBR.get(k, k.split('.')[-1])}{v:g}" if isinstance(v, (int, float)) else
                     f"{ABBR.get(k, k.split('.')[-1])}-{v}")
    return "_".join(parts)


def load_grid(path):
    with open(resolve(path), encoding="utf-8") as f:
        return yaml.safe_load(f)


def expand(grid, out_root):
    """Retorna lista de (out_dir, overrides, cond, family)."""
    jobs = []
    offs = grid.get("seed_offsets", {"conn": 0, "stim": 1000, "loc": 2000})

    def add(family, cond):
        for s in grid["seeds"]:
            ov = dict(grid.get("overrides", {}))
            ov.update(cond)
            ov.update({f"seeds.{k}": int(s) + int(o) for k, o in offs.items()})
            d = out_root / family / f"{_slug(cond)}_s{s}"
            jobs.append((d, ov, dict(cond, seed=s), family))

    keys = list(grid["axes"])
    for vals in product(*[grid["axes"][k] for k in keys]):
        add("main", dict(zip(keys, vals)))
    ctl = grid.get("controls")
    if ctl:
        ck = list(ctl["axes"])
        for vals in product(*[ctl["axes"][k] for k in ck]):
            cond = dict(ctl.get("base", {}))
            cond.update(dict(zip(ck, vals)))
            add("controls", cond)
    return jobs