"""Convergência numérica: mesma simulação com dt e dt/2 (guia §7, tabela do MVP).

Com ruído congelado os eventos de entrada são idênticos; diferenças vêm só da integração.
Em regime caótico, diferenças minúsculas crescem: por isso comparamos os trens de spikes
numa janela inicial curta e as ESTATÍSTICAS (taxas, espectro, PLV) na janela inteira.
"""

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np

from neurochaos.config import load_config, resolve
from neurochaos.jsonutil import to_jsonable
from neurochaos.runner import run_one


def spike_match(sa, ia, sb, ib, t1, tol):
    m_a = sa < t1
    hit = 0
    for t, g in zip(sa[m_a], ia[m_a]):
        tb = sb[(ib == g)]
        hit += bool(np.any(np.abs(tb - t) <= tol))
    return hit / max(1, m_a.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mvp.yaml")
    ap.add_argument("--out", default="results/convergence_dt")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--duration", type=float, default=20000.0, help="ms; estatísticas precisam de janela longa")
    a = ap.parse_args()
    cfg = load_config(a.config)
    dt = float(cfg["simulation"]["dt_ms"])
    from neurochaos.worker import parse_sets
    base = {"chaos.enabled": False, "simulation.duration_ms": a.duration}
    base.update(parse_sets(a.set))
    od = resolve(a.out)
    runs = {}
    for name, d in (("dt", dt), ("dt_half", dt / 2)):
        ov = dict(base, **{"simulation.dt_ms": d})
        out, st, sec = run_one(a.config, od / name, ov)
        print(name, d, st, f"{sec:.1f}s")
        runs[name] = out

    def load(o):
        z = np.load(o / "data.npz", allow_pickle=True)
        m = json.loads((o / "metrics.json").read_text(encoding="utf-8"))["metrics"]
        return z, m

    za, ma = load(runs["dt"])
    zb, mb = load(runs["dt_half"])
    rep = {"dt_ms": [dt, dt / 2]}
    for w in (100.0, 300.0, 1000.0):
        rep[f"spike_match_first_{int(w)}ms_tol0.5ms"] = spike_match(za["spkt"], za["spkid"], zb["spkt"], zb["spkid"], w, 0.5)
    act = {}
    for k in ("rate_E_mean", "rate_I_mean", "cv_isi_E"):
        x, y = ma["activity"][k], mb["activity"][k]
        act[k] = {"dt": x, "dt_half": y, "rel_diff": abs(x - y) / max(abs(y), 1e-12)}
    rep["activity"] = act
    for b in ("theta", "gamma"):
        pa = np.mean([v["plv"] for v in ma["plv"][b].values()])
        pb = np.mean([v["plv"] for v in mb["plv"][b].values()])
        rep[f"plv_{b}_mean"] = {"dt": pa, "dt_half": pb, "abs_diff": abs(pa - pb)}
    rep["peak_hz"] = {lab: [ma["psd"][lab]["peak_hz"], mb["psd"][lab]["peak_hz"]] for lab in ma["psd"]}
    rep["pass"] = {
        "rates_within_5pct": all(v["rel_diff"] < 0.05 for k, v in act.items() if k.startswith("rate")),
        "early_spikes_match_95pct": rep["spike_match_first_100ms_tol0.5ms"] >= 0.95,
    }
    (od / "report.json").write_text(json.dumps(to_jsonable(rep), indent=1), encoding="utf-8")
    print(json.dumps(to_jsonable(rep), indent=1))


if __name__ == "__main__":
    main()