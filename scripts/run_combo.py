"""Adendo 3 (docs/preregistro_adendo3_combinado.md): GNN + arestas confiáveis + Schaefer-400, identificação.

  python scripts/run_combo.py
"""

import hashlib
import json
import time

import _bootstrap  # noqa: F401
import numpy as np
import torch

from neurochaos import augment, gan, hopf, mica, sc_models
from neurochaos.config import resolve
from neurochaos.jsonutil import to_jsonable
from run_ident import b2_ident, edge_selection, identify, mcnemar

PARC = "schaefer400"
TR = 0.6
LAYERS = (1, 2)
QS = (0.25, 0.50, 1.0)


def main():
    t0 = time.time()
    torch.set_num_threads(1)
    out = resolve("results/identificacao/combinado")
    out.mkdir(parents=True, exist_ok=True)
    pre = resolve("docs/preregistro_adendo3_combinado.md")
    sha = hashlib.sha256(pre.read_bytes()).hexdigest()
    (out / "preregistro_sha256.txt").write_text(f"{sha}  docs/preregistro_adendo3_combinado.md\n", encoding="utf-8")
    print("adendo 3 SHA-256:", sha, "| dispositivo:", gan.device(), flush=True)

    mats = mica.load_matrices(PARC)
    subs = sorted(mats)
    labels = mats[subs[0]]["labels"]
    n = len(labels)
    iu = np.triu_indices(n, 1)
    folds = mica.read_folds()
    feat, Y, fcA, fcB, pools = {}, {}, {}, {}, {}
    rng_pool = np.random.default_rng(20261015)
    for s, d in mats.items():
        d["y"] = Y[s] = sc_models.target(d["sc"])
        feat[s] = {"y": Y[s], "fc": sc_models.upper(d["fc"]), "gd": sc_models.upper(d["gd"]),
                   "mpc": sc_models.upper(d["mpc"]), "inter": sc_models.interhemispheric(labels)}
        ts, _ = mica.load(s, PARC, "timeseries")
        h = len(ts) // 2
        fcA[s] = np.arctanh(np.clip(hopf.fc(hopf.bandpass(ts[:h], TR))[iu], -0.999, 0.999))
        fcB[s] = np.arctanh(np.clip(hopf.fc(hopf.bandpass(ts[h:], TR))[iu], -0.999, 0.999))
        P = np.arctanh(np.clip(augment.block_bootstrap_fc(ts, 10, 25, rng_pool), -0.999, 0.999))
        P[:, np.arange(n), np.arange(n)] = 0.0
        pools[s] = P.astype(np.float32)
    print(f"dados e banco de FCs prontos ({time.time() - t0:.0f}s)", flush=True)

    def train_predict(train, val, pred_subs, layers, seed):
        mu = np.mean([Y[s] for s in train], axis=0)
        b2 = sc_models.LinearMultimodal().fit([feat[s] for s in train])
        base = {s: b2.predict(feat[s]) for s in set(train) | set(val) | set(pred_subs)}
        Xtr, Ytr, Otr = gan.build_tensors(mats, train, mu, n, base)
        Xva, Yva, Ova = gan.build_tensors(mats, val, mu, n, base)
        Xp, _, _ = gan.build_tensors(mats, pred_subs, mu, n, base)
        pool = torch.tensor(np.stack([pools[s] for s in train]))
        G, info = gan.train_v2(Xtr, Ytr, Otr, Xva, Yva, Ova, fc_pool=pool, arch=f"gnn_h64_l{layers}",
                               lam_adv=0.0, lam_corr=10.0, epochs=200, seed=seed)
        dm = gan.predict(G, Xp)
        G = Xtr = Ytr = Xva = Xp = pool = None  # libera memória da GPU
        torch.cuda.empty_cache()
        return {s: base[s] + d[iu] for s, d in zip(pred_subs, dm)}, mu, info

    hits = {"combinado": {}, "B2": {}}
    chosen, inner_scores = {}, {}
    for k in sorted(set(folds.values())):
        test = [s for s in subs if folds[s] == k]
        train = [s for s in subs if folds[s] != k]
        inner = np.random.default_rng(500 + k).permutation(train)
        ifolds = [list(inner[i::4]) for i in range(4)]
        sc = {}
        for L in LAYERS:
            accs = {q: [] for q in QS}
            for ival in ifolds:
                itr = [s for s in train if s not in ival]
                pred, mu, _ = train_predict(itr, ival, ival, L, seed=0)
                for q in QS:
                    idx = edge_selection(itr, "fc_rel", q, feat, fcA, fcB)
                    h_ = identify({s: pred[s][idx] for s in ival}, {c: Y[c][idx] for c in train}, mu[idx], train)
                    accs[q].append(np.mean(list(h_.values())))
            for q in QS:
                sc["l%d_q%g" % (L, q)] = float(np.mean(accs[q]))
            print(f"fold {k} | camadas {L}: " + " ".join("q%g=%.3f" % (q, sc["l%d_q%g" % (L, q)]) for q in QS) +
                  f" ({time.time() - t0:.0f}s)", flush=True)
        best = max(((L, q) for L in LAYERS for q in QS), key=lambda c: (sc["l%d_q%g" % c], -c[0], c[1]))
        chosen[k], inner_scores[k] = "l%d_q%g" % best, sc
        rng = np.random.default_rng(1000 + k)
        val = list(rng.choice(train, size=5, replace=False))
        tr35 = [s for s in train if s not in val]
        preds = [train_predict(tr35, val, test, best[0], seed)[0] for seed in (0, 1, 2)]
        idx = edge_selection(train, "fc_rel", best[1], feat, fcA, fcB)
        mean_pred = {s: np.mean([p[s] for p in preds], axis=0)[idx] for s in test}
        mu35 = np.mean([Y[s] for s in tr35], axis=0)
        hits["combinado"].update(identify(mean_pred, {c: Y[c][idx] for c in subs}, mu35[idx], subs))
        hits["B2"].update(b2_ident(train, test, subs, np.arange(len(mu35)), feat, Y))
        print(f"fold {k}: escolhido {chosen[k]} ({time.time() - t0:.0f}s)", flush=True)

    acc = {m: float(np.mean(list(h.values()))) for m, h in hits.items()}
    prev = json.loads(resolve("results/identificacao/identificacao.json").read_text(encoding="utf-8"))
    es_hits = prev["parcs"][PARC]["hits"]["ES"]
    res = {"sha256": sha, "acc": acc, "chosen": chosen, "inner_scores": inner_scores,
           "primario_combinado_vs_B2": mcnemar(hits["combinado"], hits["B2"]),
           "secundario_combinado_vs_B2_arestas": mcnemar(hits["combinado"], es_hits),
           "acc_B2_arestas_adendo2": float(np.mean(list(es_hits.values()))),
           "hits": {m: {s: bool(v) for s, v in h.items()} for m, h in hits.items()}, "elapsed_s": time.time() - t0}
    (out / "combinado.json").write_text(json.dumps(to_jsonable(res), indent=1), encoding="utf-8")
    print("\nidentificação (Schaefer-400): combinado %.0f%% | B2 %.0f%% | B2+arestas (adendo 2) %.0f%%"
          % (100 * acc["combinado"], 100 * acc["B2"], 100 * res["acc_B2_arestas_adendo2"]))
    print("primário (McNemar vs B2):", res["primario_combinado_vs_B2"])
    print(f"tempo: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()