"""Adendo 2 (docs/preregistro_adendo2_identificacao.md): maximizar a identificação FC → SC sem inflar.

Método 1: B2 com seleção de arestas confiáveis (critério e fração escolhidos por validação interna).
Método 2: identificação em espaço comum (PCA por modalidade + CCA), componentes por validação interna.
Desfecho: acurácia de identificação (50 candidatos, acaso 2 %); McNemar exato vs B2; Holm nos 4 primários.

  python scripts/run_ident.py
"""

import hashlib
import itertools
import json
import time

import _bootstrap  # noqa: F401
import numpy as np
from scipy.stats import binomtest

from neurochaos import hopf, mica, sc_models
from neurochaos.config import resolve
from neurochaos.jsonutil import to_jsonable

TR = 0.6
PARCS = ("schaefer200", "schaefer400")
ES_GRID = list(itertools.product(("fc_rel", "sc_var"), (0.10, 0.25, 0.50, 1.0)))
CCA_GRID = [(kp, kc) for kp in (10, 20, 30) for kc in (2, 5, 10) if kc <= kp]


def identify(pred, truth_all, mu, cand):
    """pred: {s: vetor}; truth_all: {s: vetor}; candidatos = lista de sujeitos. Retorna {s: acerto(bool)}."""
    T = np.array([truth_all[c] - mu for c in cand])
    T = (T - T.mean(1, keepdims=True)) / (T.std(1, keepdims=True) + 1e-12)
    hits = {}
    for s, p in pred.items():
        q = p - mu
        q = (q - q.mean()) / (q.std() + 1e-12)
        hits[s] = cand[int(np.argmax(T @ q))] == s
    return hits


def edge_selection(train, crit, q, feat, fcA, fcB):
    m = len(feat[train[0]]["y"])
    if q >= 1.0:
        return np.arange(m)
    if crit == "fc_rel":
        A = np.array([fcA[s] for s in train])
        B = np.array([fcB[s] for s in train])
        A = (A - A.mean(0)) / (A.std(0) + 1e-12)
        B = (B - B.mean(0)) / (B.std(0) + 1e-12)
        score = (A * B).mean(0)  # correlação entre sujeitos, por aresta
    else:
        Y = np.array([feat[s]["y"] for s in train])
        present = (Y > 0).mean(0) >= 0.95
        score = np.where(present, Y.std(0) / (Y.mean(0) + 1e-12), -np.inf)
    k = max(10, int(round(q * m)))
    return np.argsort(score)[::-1][:k]


def sub(f, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in f.items()}


def b2_ident(train, test, cand, idx, feat, Y):
    m = sc_models.LinearMultimodal().fit([sub(feat[s], idx) for s in train])
    mu = np.mean([Y[s][idx] for s in train], axis=0)
    pred = {s: m.predict(sub(feat[s], idx)) for s in test}
    return identify(pred, {c: Y[c][idx] for c in cand}, mu, cand)


def cca_fit(Xtr, Ytr, kp, kc, reg=1e-3):
    xm, ym = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - xm, Ytr - ym
    kp = min(kp, len(Xtr) - 2)
    kc = min(kc, kp)
    Vx = np.linalg.svd(Xc, full_matrices=False)[2][:kp].T
    Vy = np.linalg.svd(Yc, full_matrices=False)[2][:kp].T
    Zx, Zy = Xc @ Vx, Yc @ Vy
    n = len(Zx)
    Cxx = Zx.T @ Zx / (n - 1)
    Cyy = Zy.T @ Zy / (n - 1)
    Cxy = Zx.T @ Zy / (n - 1)

    def inv_sqrt(C):
        w, V = np.linalg.eigh(C + reg * np.trace(C) / len(C) * np.eye(len(C)))
        return V @ np.diag(w ** -0.5) @ V.T

    Kx, Ky = inv_sqrt(Cxx), inv_sqrt(Cyy)
    U, S, Wt = np.linalg.svd(Kx @ Cxy @ Ky)
    Wx, Wy = Kx @ U[:, :kc], Ky @ Wt.T[:, :kc]
    return {"xm": xm, "ym": ym, "Px": Vx @ Wx, "Py": Vy @ Wy, "rho": S[:kc]}


def cca_ident(train, test, cand, kp, kc, X, Y):
    m = cca_fit(np.array([X[s] for s in train]), np.array([Y[s] for s in train]), kp, kc)
    V = np.array([(Y[c] - m["ym"]) @ m["Py"] for c in cand])
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    hits = {}
    for s in test:
        u = (X[s] - m["xm"]) @ m["Px"]
        u /= np.linalg.norm(u) + 1e-12
        hits[s] = cand[int(np.argmax(V @ u))] == s
    return hits


def mcnemar(a, b):
    """a, b: {s: acerto}. Teste exato nos pares discordantes."""
    n10 = sum(a[s] and not b[s] for s in a)
    n01 = sum(b[s] and not a[s] for s in a)
    p = float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
    return {"so_metodo": n10, "so_B2": n01, "p_exato": p}


def holm(ps):
    keys = sorted(ps, key=ps.get)
    out, prev = {}, 0.0
    for i, k in enumerate(keys):
        prev = max(prev, min(1.0, (len(keys) - i) * ps[k]))
        out[k] = prev
    return out


def main():
    t0 = time.time()
    out = resolve("results/identificacao")
    out.mkdir(parents=True, exist_ok=True)
    pre = resolve("docs/preregistro_adendo2_identificacao.md")
    sha = hashlib.sha256(pre.read_bytes()).hexdigest()
    (out / "preregistro_sha256.txt").write_text(f"{sha}  docs/preregistro_adendo2_identificacao.md\n", encoding="utf-8")
    print("adendo 2 SHA-256:", sha, flush=True)
    folds = mica.read_folds()
    res = {"sha256": sha, "parcs": {}}
    pvals = {}
    for parc in PARCS:
        mats = mica.load_matrices(parc)
        subs = sorted(mats)
        labels = mats[subs[0]]["labels"]
        n = len(labels)
        iu = np.triu_indices(n, 1)
        feat, Y, X, fcA, fcB = {}, {}, {}, {}, {}
        for s, d in mats.items():
            Y[s] = sc_models.target(d["sc"])
            fc = sc_models.upper(d["fc"])
            feat[s] = {"y": Y[s], "fc": fc, "gd": sc_models.upper(d["gd"]), "mpc": sc_models.upper(d["mpc"]),
                       "inter": sc_models.interhemispheric(labels)}
            X[s] = np.arctanh(np.clip(fc, -0.999, 0.999))
            ts, _ = mica.load(s, parc, "timeseries")
            h = len(ts) // 2
            fcA[s] = np.arctanh(np.clip(hopf.fc(hopf.bandpass(ts[:h], TR))[iu], -0.999, 0.999))
            fcB[s] = np.arctanh(np.clip(hopf.fc(hopf.bandpass(ts[h:], TR))[iu], -0.999, 0.999))
        print(f"[{parc}] dados carregados ({time.time() - t0:.0f}s)", flush=True)

        hits = {"B2": {}, "ES": {}, "CCA": {}}
        chosen = {"ES": {}, "CCA": {}}
        inner_scores = {"ES": {}, "CCA": {}}
        for k in sorted(set(folds.values())):
            test = [s for s in subs if folds[s] == k]
            train = [s for s in subs if folds[s] != k]
            rng = np.random.default_rng(500 + k)
            inner = rng.permutation(train)
            ifolds = [list(inner[i::4]) for i in range(4)]
            # --- seleção interna
            es_sc, cca_sc = {}, {}
            for crit, q in ES_GRID:
                acc = []
                for ival in ifolds:
                    itr = [s for s in train if s not in ival]
                    idx = edge_selection(itr, crit, q, feat, fcA, fcB)
                    acc.append(np.mean(list(b2_ident(itr, ival, train, idx, feat, Y).values())))
                es_sc[f"{crit}_q{q:g}"] = float(np.mean(acc))
            for kp, kc in CCA_GRID:
                acc = []
                for ival in ifolds:
                    itr = [s for s in train if s not in ival]
                    acc.append(np.mean(list(cca_ident(itr, ival, train, kp, kc, X, Y).values())))
                cca_sc[f"pca{kp}_cca{kc}"] = float(np.mean(acc))
            best_es = max(ES_GRID, key=lambda c: (es_sc[f"{c[0]}_q{c[1]:g}"], c[1], c[0] == "fc_rel"))
            best_cca = max(CCA_GRID, key=lambda c: (cca_sc[f"pca{c[0]}_cca{c[1]}"], -c[0], -c[1]))
            chosen["ES"][k], chosen["CCA"][k] = f"{best_es[0]}_q{best_es[1]:g}", f"pca{best_cca[0]}_cca{best_cca[1]}"
            inner_scores["ES"][k], inner_scores["CCA"][k] = es_sc, cca_sc
            # --- teste externo (candidatos = os 50)
            full = np.arange(len(Y[subs[0]]))
            hits["B2"].update(b2_ident(train, test, subs, full, feat, Y))
            idx = edge_selection(train, best_es[0], best_es[1], feat, fcA, fcB)
            hits["ES"].update(b2_ident(train, test, subs, idx, feat, Y))
            hits["CCA"].update(cca_ident(train, test, subs, best_cca[0], best_cca[1], X, Y))
            print(f"[{parc}] fold {k}: ES={chosen['ES'][k]} CCA={chosen['CCA'][k]} ({time.time() - t0:.0f}s)", flush=True)

        r = {"acc": {m: float(np.mean(list(h.values()))) for m, h in hits.items()}, "chosen": chosen,
             "inner_scores": inner_scores, "hits": {m: {s: bool(v) for s, v in h.items()} for m, h in hits.items()}}
        for m in ("ES", "CCA"):
            r[f"mcnemar_{m}_vs_B2"] = mcnemar(hits[m], hits["B2"])
            pvals[f"{parc}_{m}"] = r[f"mcnemar_{m}_vs_B2"]["p_exato"]
        res["parcs"][parc] = r
        print(f"[{parc}] identificação: B2 {r['acc']['B2']:.0%} | ES {r['acc']['ES']:.0%} | CCA {r['acc']['CCA']:.0%}",
              flush=True)
    res["holm"] = holm(pvals)
    res["elapsed_s"] = time.time() - t0
    (out / "identificacao.json").write_text(json.dumps(to_jsonable(res), indent=1), encoding="utf-8")
    print("Holm:", {k: round(v, 4) for k, v in res["holm"].items()})
    print(f"tempo: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()