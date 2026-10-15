"""Calibração: F–I → reobase → varredura do peso inter-regional → K_base (guia §6–§7).

Escreve configs/calibration_mvp.json, aplicado automaticamente por load_config().
Os critérios (taxas, fração silenciosa, saturação) estão em `calibration.criteria` do YAML.
"""

import argparse
import datetime
import json
import subprocess
import sys

import _bootstrap  # noqa: F401
import numpy as np

from neurochaos.config import ROOT, load_config, resolve
from neurochaos.runner import run_jobs


def meets(act, crit):
    ok = crit["rate_E_mean"][0] <= act["rate_E_mean"] <= crit["rate_E_mean"][1]
    ok &= crit["rate_I_mean"][0] <= act["rate_I_mean"] <= crit["rate_I_mean"][1]
    ok &= act["rate_E_max"] <= crit["rate_E_max_le"]
    ok &= act["frac_silent_E"] <= crit["frac_silent_E_le"]
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mvp.yaml")
    ap.add_argument("--out", default="results/calibration")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, use_calibration=False)
    od = resolve(a.out)

    cal = cfg["calibration"]
    subprocess.run([sys.executable, str(ROOT / "scripts" / "fi_curve.py"), "--config", a.config, "--out", str(od),
                    "--amax", str(cal.get("fi_amax_nA", 0.30)), "--step", str(cal.get("fi_step_nA", 0.0025))],
                   check=True, cwd=ROOT)
    fi = json.loads((od / "fi_curve.json").read_text(encoding="utf-8"))
    rheo = {ct: fi[ct]["rheobase_repetitive_nA"] for ct in ("E", "I")}

    jobs = []
    for w in cal["sweep_K_base_uS"]:
        for s in cal["seeds"]:
            ov = {"coupling.K": 1.0, "coupling.K_base_uS": w, "chaos.enabled": False,
                  "stimulus.theta.amp_nA": 0.0, "stimulus.gamma.amp_nA": 0.0,
                  "connectome.topology": "distance", "background.rheobase_nA": rheo,
                  "seeds.conn": s, "seeds.stim": 1000 + s, "seeds.loc": 2000 + s}
            jobs.append((od / "sweep" / f"w{w:g}_s{s}", ov))
    run_jobs(a.config, jobs, n_workers=a.workers, no_arrays=True)

    rows = []
    for d, ov in jobs:
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))["metrics"]
        act = m["activity"]
        plv = np.mean([v["plv"] for v in m["plv"]["gamma"].values()])
        rows.append({"w_uS": ov["coupling.K_base_uS"], "seed": ov["seeds.conn"],
                     "rate_E_mean": act["rate_E_mean"], "rate_I_mean": act["rate_I_mean"],
                     "rate_E_max": act["rate_E_max"], "frac_silent_E": act["frac_silent_E"],
                     "plv_gamma_mean": float(plv), "valid": meets(act, cal["criteria"])})
    ws = sorted(set(r["w_uS"] for r in rows))
    valid_w = [w for w in ws if all(r["valid"] for r in rows if r["w_uS"] == w)]
    print("\n w (µS)    rate_E   rate_I   PLVγ   válido")
    for w in ws:
        rr = [r for r in rows if r["w_uS"] == w]
        print(f" {w:<9g} {np.mean([r['rate_E_mean'] for r in rr]):7.1f} {np.mean([r['rate_I_mean'] for r in rr]):8.1f}"
              f" {np.mean([r['plv_gamma_mean'] for r in rr]):6.3f}   {all(r['valid'] for r in rr)}")
    positive = [w for w in valid_w if w > 0]
    if not positive:
        raise SystemExit("Nenhum peso inter-regional > 0 satisfaz os critérios; revise parâmetros locais/fundo.")
    # Maior peso válido contíguo a partir do menor: evita escolher um ponto isolado.
    contiguous = []
    for w in ws:
        if w in valid_w:
            contiguous.append(w)
        elif contiguous:
            break
    w_max = max(x for x in contiguous if x > 0)
    K_base = w_max / float(cal["K_max_in_grid"])
    print(f"\nreobase: {rheo}  |  maior peso válido: {w_max:g} µS  ->  K_base = {K_base:g} µS")

    out = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "config_updates": {"background": {"rheobase_nA": rheo}, "coupling": {"K_base_uS": K_base}},
        "fi_curve": {ct: {k: fi[ct][k] for k in fi[ct] if k != "rates_hz"} for ct in ("E", "I")},
        "criteria": cal["criteria"], "sweep": rows, "max_valid_w_uS": w_max,
        "K_max_in_grid": cal["K_max_in_grid"],
        "note": "Calibração do MVP com duração de depuração; repetir com a duração científica antes de concluir.",
    }
    dst = resolve(cfg["calibration_file"])
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("escrito:", dst)


if __name__ == "__main__":
    main()