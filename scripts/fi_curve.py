"""Curvas F–I e reobase de cada tipo celular, ANTES da rede (guia §7, ajustes de excitabilidade).

Cada célula recebe um degrau de corrente diferente; a taxa é medida em regime estacionário.
Também mede duração do PA (meia amplitude) e potencial de repouso.
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from neurochaos.cells import cell_rule
from neurochaos.config import load_config
from neurochaos.jsonutil import to_jsonable
from neurochaos import signals as sg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mvp.yaml")
    ap.add_argument("--out", default="results/calibration")
    ap.add_argument("--amax", type=float, default=0.30)
    ap.add_argument("--step", type=float, default=0.0025)
    a = ap.parse_args()

    from netpyne import sim, specs
    from neuron import h

    cfg = load_config(a.config, use_calibration=False)
    amps = np.round(np.arange(0.0, a.amax + 1e-12, a.step), 6)
    delay, dur, settle = 200.0, 2000.0, 500.0

    net = specs.NetParams()
    net.defaultThreshold = cfg["simulation"]["spike_threshold_mv"]
    for ct in ("E", "I"):
        net.cellParams[ct] = cell_rule(cfg["cells"][ct])
        net.popParams[ct] = {"cellType": ct, "numCells": len(amps)}
    net.stimSourceParams["step"] = {"type": "IClamp", "del": delay, "dur": dur, "amp": 0.0}
    net.stimTargetParams["step"] = {"source": "step", "conds": {"cellType": ["E", "I"]}, "sec": "soma", "loc": 0.5}
    sc = specs.SimConfig()
    sc.duration = delay + dur + 100
    sc.dt = cfg["simulation"]["dt_ms"]
    sc.hParams = {"celsius": cfg["simulation"]["celsius"], "v_init": cfg["simulation"]["v_init_mv"]}
    sc.recordTraces = {}
    sc.verbose = False
    sim.create(netParams=net, simConfig=sc, output=False)

    cells = sorted(sim.net.cells, key=lambda c: c.gid)
    vrec = {}
    t = h.Vector().record(h._ref_t, 0.05)
    for ct in ("E", "I"):
        cl = [c for c in cells if c.tags["cellType"] == ct]
        for k, c in enumerate(cl):
            c.stims[0]["hObj"].amp = float(amps[k])
            vrec[c.gid] = h.Vector().record(c.secs["soma"]["hObj"](0.5)._ref_v, 0.05)
    sim.runSim()
    spkt, spkid = np.array(sim.simData["spkt"]), np.array(sim.simData["spkid"]).astype(int)
    tt = np.array(t)

    out = {"amps_nA": amps.tolist(), "window_ms": [delay + settle, delay + dur]}
    for ct in ("E", "I"):
        cl = [c for c in cells if c.tags["cellType"] == ct]
        rates, first_spike = [], []
        for c in cl:
            st = spkt[spkid == c.gid]
            r = np.sum((st >= delay + settle) & (st < delay + dur)) / ((dur - settle) / 1000.0)
            rates.append(float(r))
            first_spike.append(bool(np.any(st >= delay)))
        rates = np.array(rates)
        rep = np.where(rates > 0)[0]
        single = np.where(first_spike)[0]
        rheo = float(amps[rep[0]]) if len(rep) else float("nan")
        k2 = int(np.argmin(np.abs(amps - 1.5 * rheo))) if len(rep) else len(amps) - 1
        v = np.array(vrec[cl[k2].gid])
        m = (tt >= delay + settle) & (tt < delay + dur)
        width, _ = sg.spike_shape(tt[m], v[m])
        st2 = np.sort(spkt[(spkid == cl[k2].gid) & (spkt >= delay) & (spkt < delay + dur)])
        isi = np.diff(st2)
        adapt = float(np.mean((isi[1:] - isi[:-1]) / (isi[1:] + isi[:-1]))) if len(isi) > 2 else float("nan")
        first_isi = float(isi[0]) if len(isi) else float("nan")
        last_isi = float(isi[-1]) if len(isi) else float("nan")
        v_rest = float(np.array(vrec[cl[0].gid])[(tt > delay - 50) & (tt < delay)].mean())
        out[ct] = {"rates_hz": rates.tolist(),
                   "rheobase_repetitive_nA": rheo,
                   "threshold_single_spike_nA": float(amps[single[0]]) if len(single) else float("nan"),
                   "min_repetitive_rate_hz": float(rates[rep[0]]) if len(rep) else float("nan"),
                   "ap_halfwidth_ms": width, "v_rest_mV": v_rest,
                   "adaptation_index_1.5x": adapt, "first_isi_ms_1.5x": first_isi, "last_isi_ms_1.5x": last_isi}
        print(f"{ct}: reobase (repetitivo) = {rheo:.4f} nA; f_min = {out[ct]['min_repetitive_rate_hz']:.1f} Hz; "
              f"limiar 1 PA = {out[ct]['threshold_single_spike_nA']:.4f} nA; V_rest = {v_rest:.1f} mV; "
              f"PA½ = {width:.2f} ms; adaptação (1,5×reobase) = {adapt:.3f} (ISI {first_isi:.1f}→{last_isi:.1f} ms)")

    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    (od / "fi_curve.json").write_text(json.dumps(to_jsonable(out), indent=1), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for ct, mk in (("E", "o"), ("I", "s")):
            ax.plot(amps, out[ct]["rates_hz"], mk + "-", ms=3, label=ct)
        ax.set_xlabel("I degrau (nA)")
        ax.set_ylabel("taxa estacionária (Hz)")
        ax.set_title(f"F–I, HH {cfg['simulation']['celsius']} °C")
        ax.legend()
        fig.tight_layout()
        fig.savefig(od / "fi_curve.png", dpi=130)
    except Exception as e:
        print("figura não gerada:", e)


if __name__ == "__main__":
    main()