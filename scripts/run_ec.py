"""Adendo D: conectividade efetiva individual pelo Hopf linearizado. Ajuste na FC da 1ª metade do BOLD;
avaliação na 2ª metade (FC) e contra a SC da difusão do próprio sujeito (estrutura).

  python scripts/run_ec.py
"""

import csv
import json
import time

import _bootstrap  # noqa: F401
import numpy as np
from scipy.stats import binomtest, wilcoxon

from neurochaos import ec, hopf, mica, sc_metrics, sc_models
from neurochaos.config import resolve
from neurochaos.jsonutil import to_jsonable

PARC = "schaefer200"
TR = 0.6


def to_target(M):
    iu = np.triu_indices(len(M), 1)
    v = np.maximum(M[iu], 0.0)
    return np.log1p(v / v.mean())


def main():
    t0 = time.time()
    out = resolve("results/escada")
    out.mkdir(parents=True, exist_ok=True)
    mats = mica.load_matrices(PARC)
    subs = sorted(mats)
    n = len(mats[subs[0]]["labels"])
    labels = mats[subs[0]]["labels"]
    iu = np.triu_indices(n, 1)
    folds = mica.read_folds()
    Y = {s: sc_models.target(d["sc"]) for s, d in mats.items()}
    feat = {s: {"y": Y[s], "fc": sc_models.upper(d["fc"]), "gd": sc_models.upper(d["gd"]),
                "mpc": sc_models.upper(d["mpc"]), "inter": sc_models.interhemispheric(labels)} for s, d in mats.items()}

    preds_ec, preds_b2, mus, rows = {}, {}, {}, []
    for k in sorted(set(folds.values())):
        train = [s for s in subs if folds[s] != k]
        test = [s for s in subs if folds[s] == k]
        mu = np.mean([Y[s] for s in train], axis=0)
        grp = np.mean([mats[s]["sc"] for s in train], axis=0)
        np.fill_diagonal(grp, 0.0)
        C0 = 0.2 * grp / grp.max()
        mask = (grp > 0).astype(float)
        b2 = sc_models.LinearMultimodal().fit([feat[s] for s in train])
        for s in test:
            ts, _ = mica.load(s, PARC, "timeseries")
            h = len(ts) // 2
            fcA, fcB = hopf.fc(hopf.bandpass(ts[:h], TR)), hopf.fc(hopf.bandpass(ts[h:], TR))
            freqs = hopf.intrinsic_freqs(ts[:h], TR)
            F0 = ec.model_fc(C0, freqs)
            EC, hist = ec.fit_ec(C0, fcA, freqs, mask, log_every=50)
            F1 = ec.model_fc(EC, freqs)
            preds_ec[s] = to_target(EC)
            # B2 com a FC só da 1ª metade (mesma informação que a EC recebeu)
            preds_b2[s] = b2.predict(dict(feat[s], fc=fcA[iu]))
            mus[s] = mu
            r = {"subject": s, "fold": k,
                 "fc_inicial_x_B": float(np.corrcoef(F0[iu], fcB[iu])[0, 1]),
                 "fc_EC_x_A": hist[-1][2], "iter_escolhida": hist[-1][1], "fc_EC_x_B": float(np.corrcoef(F1[iu], fcB[iu])[0, 1]),
                 "teto_A_x_B": float(np.corrcoef(fcA[iu], fcB[iu])[0, 1])}
            rows.append(r)
            print(f"{s}: FC_modelo×B inicial={r['fc_inicial_x_B']:.3f} → EC={r['fc_EC_x_B']:.3f} "
                  f"(ajuste A={r['fc_EC_x_A']:.3f}, teto={r['teto_A_x_B']:.3f})  [{time.time() - t0:.0f}s]", flush=True)

    res = {"fc": {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("subject", "fold")}, "models": {},
           "adendo_sha256_com_emenda": __import__("hashlib").sha256(
               resolve("docs/preregistro_adendo_modelos.md").read_bytes()).hexdigest()}
    met = {}
    for name, P in (("EC", preds_ec), ("B2_fcA", preds_b2)):
        met[name] = {s: sc_metrics.edge_metrics(P[s], Y[s], mus[s], n) for s in subs}
        idn = sc_metrics.identification(P, Y, mus)
        res["models"][name] = {mt: float(np.mean([met[name][s][mt] for s in subs]))
                               for mt in ("pearson", "pearson_desvio", "auc_top20")}
        res["models"][name]["ident_accuracy"] = idn["accuracy"]
        res["models"][name]["ident_p"] = float(binomtest(int(round(idn["accuracy"] * 50)), 50, 1 / 50,
                                                         alternative="greater").pvalue)
    x = np.array([met["EC"][s]["pearson_desvio"] for s in subs])
    y = np.array([met["B2_fcA"][s]["pearson_desvio"] for s in subs])
    res["teste_D"] = {"mean_EC": float(x.mean()), "mean_B2": float(y.mean()), "n_EC_maior": int(np.sum(x > y)),
                      "p_wilcoxon": float(wilcoxon(x, y).pvalue)}
    res["elapsed_s"] = time.time() - t0
    with open(out / "ec_per_subject.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "ec.json").write_text(json.dumps(to_jsonable(res), indent=1), encoding="utf-8")
    print("\nFC (médias):", {k: round(v, 3) for k, v in res["fc"].items()})
    for nm, v in res["models"].items():
        print(f"{nm:7s} r={v['pearson']:.3f} r_desvio={v['pearson_desvio']:.4f} ident={v['ident_accuracy']:.2f} "
              f"(p={v['ident_p']:.2g})")
    print("D) EC vs B2:", res["teste_D"])
    print(f"tempo: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()