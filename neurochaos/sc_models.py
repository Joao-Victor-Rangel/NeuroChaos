"""Modelos FC/anatomia → SC (conectividade estrutural) para o plano v2 (§3.2).

Alvo: y = log1p(SC / média(SC)) no triângulo superior — normalização por sujeito remove a escala
global da tractografia e o log comprime a cauda pesada das contagens.

Entradas permitidas (o que o paciente "tem"): FC (fMRI), GD (geodésica cortical, do T1) e MPC
(microestrutura, qT1). O comprimento das fibras (edgeLength) vem da DIFUSÃO e NUNCA é entrada:
seria vazamento do gabarito.
"""

import re

import numpy as np


def target(sc):
    iu = np.triu_indices(len(sc), 1)
    v = sc[iu]
    return np.log1p(v / v.mean())


def upper(M):
    return M[np.triu_indices(len(M), 1)]


def interhemispheric(labels):
    """1 para arestas entre hemisférios (rótulos Schaefer "_LH_"/"_RH_", aparc "-lh-"/"-rh-")."""
    left = np.array([bool(re.search(r"(_LH_|-lh-|^Left)", l)) for l in labels])
    right = np.array([bool(re.search(r"(_RH_|-rh-|^Right)", l)) for l in labels])
    if not np.all(left ^ right):
        raise ValueError("hemisfério indefinido para algum rótulo")
    i, j = np.triu_indices(len(labels), 1)
    return (left[i] != left[j]).astype(float)


class GroupMean:
    """B0 — média do grupo de treino: nenhuma informação individual."""
    name = "B0_media_grupo"

    def fit(self, train):
        self.mu = np.mean([s["y"] for s in train], axis=0)
        return self

    def predict(self, s):
        return self.mu


class Distance:
    """B1 — decaimento com a distância geodésica DO PRÓPRIO sujeito (anatomia do T1)."""
    name = "B1_distancia"

    def _X(self, s):
        g = s["gd"]
        return np.column_stack([np.ones_like(g), g, np.log1p(g), s["inter"], s["inter"] * g])

    def fit(self, train):
        X = np.vstack([self._X(s) for s in train])
        y = np.concatenate([s["y"] for s in train])
        self.w = np.linalg.lstsq(X, y, rcond=None)[0]
        return self

    def predict(self, s):
        return self._X(s) @ self.w


class LinearMultimodal:
    """B2 — regressão linear agrupada: prior do grupo + FC + GD + MPC do sujeito."""
    name = "B2_linear_multimodal"

    def __init__(self, use_fc=True):
        self.use_fc = use_fc
        if not use_fc:
            self.name = "B2_sem_FC"

    def _X(self, s):
        cols = [np.ones_like(self.mu), self.mu, s["gd"], np.log1p(s["gd"]), s["mpc"], s["inter"]]
        if self.use_fc:
            fc = s["fc"]
            cols += [fc, np.arctanh(np.clip(fc, -0.999, 0.999)), fc * self.mu]
        return np.column_stack(cols)

    def fit(self, train):
        self.mu = np.mean([s["y"] for s in train], axis=0)
        X = np.vstack([self._X(s) for s in train])
        y = np.concatenate([s["y"] for s in train])
        self.w = np.linalg.lstsq(X, y, rcond=None)[0]
        return self

    def predict(self, s):
        return self._X(s) @ self.w


class KernelRidgeFC:
    """B3 — ridge entre sujeitos: desvios da SC previstos pelos desvios da FC (kernel linear).

    Com ~40 sujeitos de treino é, na prática, uma média ponderada das SC de sujeitos com FC
    parecida. λ escolhido por validação interna (k-fold) dentro do treino.
    """
    name = "B3_ridge_FC"

    def __init__(self, lambdas=(1e-2, 1e-1, 1, 10, 100, 1e3), inner_folds=4):
        self.lambdas = lambdas
        self.inner = inner_folds

    def _fit(self, F, Y, lam):
        fm, ym = F.mean(0), Y.mean(0)
        Fc, Yc = F - fm, Y - ym
        K = Fc @ Fc.T
        K /= np.trace(K) / len(K)
        alpha = np.linalg.solve(K + lam * np.eye(len(K)), Yc)
        return fm, ym, Fc, alpha, np.trace(Fc @ Fc.T) / len(Fc)

    def _pred(self, model, f):
        fm, ym, Fc, alpha, scale = model
        k = Fc @ (f - fm) / scale
        return ym + k @ alpha

    def fit(self, train):
        F = np.array([np.arctanh(np.clip(s["fc"], -0.999, 0.999)) for s in train])
        Y = np.array([s["y"] for s in train])
        idx = np.arange(len(train))
        best, best_err = None, np.inf
        for lam in self.lambdas:
            err = 0.0
            for k in range(self.inner):
                te = idx[k::self.inner]
                tr = np.setdiff1d(idx, te)
                m = self._fit(F[tr], Y[tr], lam)
                err += sum(np.mean((self._pred(m, F[i]) - Y[i]) ** 2) for i in te)
            if err < best_err:
                best, best_err = lam, err
        self.lam = best
        self.model = self._fit(F, Y, best)
        return self

    def predict(self, s):
        return self._pred(self.model, np.arctanh(np.clip(s["fc"], -0.999, 0.999)))