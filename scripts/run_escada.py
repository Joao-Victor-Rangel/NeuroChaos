"""Adendo do pré-registro (docs/preregistro_adendo_modelos.md): A) PLS de kernel aninhado; C) curva de aprendizado.

  python scripts/run_escada.py
"""

import hashlib
import json
import time

import _bootstrap  # noqa: F401
import numpy as np
from scipy.stats import binomtest, wilcoxon

from neurochaos import mica, sc_metrics, sc_models
from neurochaos.config import resolve
from neurochaos.jsonutil import to_jsonable
from neurochaos.pls import KernelPLS

PARC = "schaefer200"
COMPS = (1, 2, 3, 5, 10)


def main():
    out = resolve("results/escada")
    out.mkdir(parents=True, exist_ok=True)
    pre = resolve("docs/preregistro_adendo_modelos.md")
    sha = hashlib.sha256(pre.read_bytes()).hexdigest()
    (out / "preregistro_adendo_sha256.txt").write_text(f"{sha}  docs/preregistro_adendo_modelos.md\n", encoding="utf-8")
    print("adendo SHA-256:", sha, flush=True)
    t0 = time.time()

    mats = mica.load_matrices(PARC)
    subs = sorted(mats)
    n = len(mats[subs[0]]["labels"])
    labels = mats[subs[0]]["labels"]
    folds = mica.read_folds()
    feat, X, Y = {}, {}, {}
    for s, d in mats.items():
        y = sc_models.target(d["sc"])
        fc, gd, mpc = sc_models.upper(d["fc"]), sc_models.upper(d["gd"]), sc_models.upper(d["mpc"])
        feat[s] = {"y": y, "fc": fc, "gd": gd, "mpc": mpc, "inter": sc_models.interhemispheric(labels)}
        X[s] = np.concatenate([np.arctanh(np.clip(fc, -0.999, 0.999)), gd, mpc])
        Y[s] = y

    def pls_fit_predict(train, test, k):
        m = KernelPLS(k).fit(np.array([X[s] for s in train]), np.array([Y[s] for s in train]))
        return {s: p for s, p in zip(test, m.predict(np.array([X[s] for s in test])))}

    def r_dev(pred, true, mu):
        a, b = pred - mu, true - mu
        return float(np.corrcoef(a, b)[0, 1])

    # ---------------- A) PLS aninhado
    preds_pls, preds_b2, mus, chosen, inner_all = {}, {}, {}, {}, {}
    for k in sorted(set(folds.values())):
        test = [s for s in subs if folds[s] == k]
        train = [s for s in subs if folds[s] != k]
        rng = np.random.default_rng(500 + k)  # mesmos folds internos da busca da GAN
        inner = rng.permutation(train)
        inner_folds = [list(inner[i::4]) for i in range(4)]
        scores = {}
        for c in COMPS:
            sc = []
            for ival in inner_folds:
                itr = [s for s in train if s not in ival]
                mu_i = np.mean([Y[s] for s in itr], axis=0)
                p = pls_fit_predict(itr, ival, c)
                sc.append(np.mean([r_dev(p[s], Y[s], mu_i) for s in ival]))
            scores[c] = float(np.mean(sc))
        best = max(COMPS, key=lambda c: (scores[c], -c))  # empate → menos componentes
        chosen[k], inner_all[k] = best, scores
        mu = np.mean([Y[s] for s in train], axis=0)
        preds_pls.update(pls_fit_predict(train, test, best))
        b2 = sc_models.LinearMultimodal().fit([feat[s] for s in train])
        for s in test:
            preds_b2[s] = b2.predict(feat[s])
            mus[s] = mu
        print(f"fold {k}: escores internos {scores} → {best} componentes ({time.time() - t0:.0f}s)", flush=True)

    res = {"adendo_sha256": sha, "pls_chosen": chosen, "pls_inner": inner_all, "models": {}}
    rows = {}
    for name, P in (("PLS", preds_pls), ("B2", preds_b2)):
        rows[name] = {s: sc_metrics.edge_metrics(P[s], Y[s], mus[s], n) for s in subs}
        idn = sc_metrics.identification(P, Y, mus)
        res["models"][name] = {mt: float(np.mean([rows[name][s][mt] for s in subs]))
                               for mt in ("pearson", "pearson_desvio", "auc_top20")}
        res["models"][name]["ident_accuracy"] = idn["accuracy"]
        res["models"][name]["ident_p"] = float(binomtest(int(round(idn["accuracy"] * 50)), 50, 1 / 50,
                                                         alternative="greater").pvalue)
    x = np.array([rows["PLS"][s]["pearson_desvio"] for s in subs])
    y = np.array([rows["B2"][s]["pearson_desvio"] for s in subs])
    res["teste_A"] = {"mean_PLS": float(x.mean()), "mean_B2": float(y.mean()), "n_PLS_maior": int(np.sum(x > y)),
                      "p_wilcoxon": float(wilcoxon(x, y).pvalue)}
    print("A) PLS vs B2:", res["teste_A"], flush=True)
    for nm, v in res["models"].items():
        print(f"   {nm}: r_desvio={v['pearson_desvio']:.4f} ident={v['ident_accuracy']:.2f} (p={v['ident_p']:.2g})")

    # ---------------- C) curva de aprendizado
    k_curve = int(np.bincount(list(chosen.values())).argmax())  # nº de componentes mais escolhido na validação interna
    curve = {"pls_components": k_curve, "points": []}
    for N in (10, 20, 30, 40):
        for draw in range(5 if N < 40 else 1):
            vals = {"B2": [], "PLS": []}
            for k in sorted(set(folds.values())):
                test = [s for s in subs if folds[s] == k]
                train_all = [s for s in subs if folds[s] != k]
                rng = np.random.default_rng(10000 + 100 * N + 10 * draw + k)
                train = list(rng.choice(train_all, size=N, replace=False))
                mu = np.mean([Y[s] for s in train], axis=0)
                b2 = sc_models.LinearMultimodal().fit([feat[s] for s in train])
                pp = pls_fit_predict(train, test, min(k_curve, N - 1))
                for s in test:
                    vals["B2"].append(r_dev(b2.predict(feat[s]), Y[s], mu))
                    vals["PLS"].append(r_dev(pp[s], Y[s], mu))
            curve["points"].append({"N": N, "draw": draw, "B2": float(np.mean(vals["B2"])),
                                    "PLS": float(np.mean(vals["PLS"]))})
        pts = [p for p in curve["points"] if p["N"] == N]
        print(f"C) N={N}: B2 r_desvio={np.mean([p['B2'] for p in pts]):.4f}  PLS={np.mean([p['PLS'] for p in pts]):.4f}",
              flush=True)
    res["learning_curve"] = curve
    res["elapsed_s"] = time.time() - t0
    (out / "pls_curva.json").write_text(json.dumps(to_jsonable(res), indent=1), encoding="utf-8")
    print(f"tempo: {time.time() - t0:.0f}s; saídas em {out}")


if __name__ == "__main__":
    main()