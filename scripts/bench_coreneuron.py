"""NEURON × CoreNEURON na rede cortical: mesmos resultados? quanto mais rápido?

Pré-requisito: mecanismos compilados com suporte a CoreNEURON na raiz do projeto:
  PATH=~/neuron-gcc-wrapper:$PATH nrnivmodl -coreneuron mod

Compara, com a MESMA realização de ruído (Random123): trens de spikes, taxas e tempo de simulação,
sem estímulo e com theta (testa Vector.play no CoreNEURON).
"""

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np

from neurochaos.config import resolve
from neurochaos.jsonutil import to_jsonable
from neurochaos.runner import run_one


def spikes_match(a, b, tol=0.1):
    """Fração dos spikes de a com par em b (mesmo gid, |Δt| ≤ tol ms)."""
    hit = 0
    for g in np.unique(a["spkid"]):
        ta, tb = np.sort(a["spkt"][a["spkid"] == g]), np.sort(b["spkt"][b["spkid"] == g])
        if len(tb):
            idx = np.clip(np.searchsorted(tb, ta), 1, len(tb) - 1)
            d = np.minimum(np.abs(tb[idx] - ta), np.abs(tb[idx - 1] - ta))
            hit += int(np.sum(d <= tol))
    return hit / max(1, len(a["spkt"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cortical_atlas.yaml")
    ap.add_argument("--duration", type=float, default=3000.0)
    a = ap.parse_args()
    od = resolve("results/bench_coreneuron")
    base = {"simulation.duration_ms": a.duration, "simulation.record_all_v": False, "simulation.save_traces": False,
            "coupling.K": 2.0, "connectome.atlas_dir": "data/connectomes/mica_aparc/sub-HC006",
            "simulation.random123": True}
    rep = {}
    for stim in (0.0, 0.1):
        res = {}
        for eng in ("neuron", "coreneuron"):
            ov = dict(base, **{"stimulus.theta.amp_nA": stim, "simulation.coreneuron": eng == "coreneuron"})
            d, st, sec = run_one(a.config, od / f"{eng}_th{stim:g}", ov, no_arrays=False)
            if st != "ok":
                print(f"{eng} th={stim}: ERRO — ver {d}/log.txt e error.txt")
                res[eng] = None
                continue
            m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
            z = np.load(d / "data.npz", allow_pickle=True)
            res[eng] = {"spkt": z["spkt"], "spkid": z["spkid"], "rate_E": m["metrics"]["activity"]["rate_E_mean"],
                        "rate_I": m["metrics"]["activity"]["rate_I_mean"], "t_sim": m["meta"]["timing_s"]["main_sim"]}
            print(f"{eng:10s} th={stim:g}: E={res[eng]['rate_E']:.2f} Hz  I={res[eng]['rate_I']:.2f} Hz  "
                  f"sim={res[eng]['t_sim']:.1f}s  spikes={len(res[eng]['spkt'])}")
        if res.get("neuron") and res.get("coreneuron"):
            n, c = res["neuron"], res["coreneuron"]
            rep[f"theta_{stim:g}"] = {
                "speedup": n["t_sim"] / c["t_sim"],
                "spike_match_neuron_in_core": spikes_match(n, c),
                "spike_match_core_in_neuron": spikes_match(c, n),
                "rate_E": [n["rate_E"], c["rate_E"]], "rate_I": [n["rate_I"], c["rate_I"]],
                "n_spikes": [len(n["spkt"]), len(c["spkt"])]}
            r = rep[f"theta_{stim:g}"]
            print(f"  -> aceleração {r['speedup']:.2f}x | spikes coincidentes (±0,1 ms): "
                  f"{r['spike_match_neuron_in_core']:.4f} / {r['spike_match_core_in_neuron']:.4f}")
    (od / "report.json").write_text(json.dumps(to_jsonable(rep), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()