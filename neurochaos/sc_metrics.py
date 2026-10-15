"""Métricas de concordância SC prevista × SC da difusão do mesmo sujeito (plano v2 §3.3)."""

import numpy as np
from scipy.stats import spearmanr


def auc(score, positive):
    """AUC de Mann–Whitney (empates com meio ponto)."""
    from scipy.stats import rankdata
    r = rankdata(score)
    n1 = positive.sum()
    n0 = len(positive) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def edge_metrics(pred, true, mu_train, n_nodes):
    """pred/true/mu_train: vetores do triângulo superior (escala do alvo)."""
    out = {
        "pearson": float(np.corrcoef(pred, true)[0, 1]),
        "spearman": float(spearmanr(pred, true).correlation),
        "mse": float(np.mean((pred - true) ** 2)),
    }
    strong = true >= np.quantile(true, 0.8)
    out["auc_top20"] = auc(pred, strong)
    # Parte individual: desvio em relação à média do grupo de treino.
    dp, dt = pred - mu_train, true - mu_train
    out["pearson_desvio"] = float(np.corrcoef(dp, dt)[0, 1]) if dp.std() > 1e-12 else float("nan")
    # Força nodal (soma dos pesos por nó).
    M_p, M_t = np.zeros((n_nodes, n_nodes)), np.zeros((n_nodes, n_nodes))
    iu = np.triu_indices(n_nodes, 1)
    M_p[iu], M_t[iu] = pred, true
    sp, st = (M_p + M_p.T).sum(1), (M_t + M_t.T).sum(1)
    out["pearson_forca_nodal"] = float(np.corrcoef(sp, st)[0, 1])
    return out


def identification(pred_by_sub, true_by_sub, mu_by_sub):
    """Acurácia de identificação: a SC prevista de i é mais parecida com a SC real de i?

    Usa desvios em relação à média do grupo de treino do fold de i (senão a estrutura comum,
    idêntica para todos, domina a correlação). Chance = 1 / n_sujeitos.
    """
    subs = sorted(pred_by_sub)
    T = np.array([true_by_sub[s] for s in subs])
    hits, diffs = [], []
    C = np.zeros((len(subs), len(subs)))
    for a, s in enumerate(subs):
        mu = mu_by_sub[s]
        p = pred_by_sub[s] - mu
        if p.std() < 1e-12:
            C[a] = np.nan
            hits.append(np.nan)
            continue
        D = T - mu
        c = np.array([np.corrcoef(p, d)[0, 1] for d in D])
        C[a] = c
        hits.append(float(np.argmax(c) == a))
        diffs.append(float(c[a] - np.delete(c, a).mean()))
    return {"accuracy": float(np.nanmean(hits)) if not np.all(np.isnan(hits)) else float("nan"),
            "chance": 1.0 / len(subs),
            "differential_identifiability": float(np.mean(diffs)) if diffs else float("nan"),
            "matrix": C, "subjects": subs}