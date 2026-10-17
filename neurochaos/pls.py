"""PLS de kernel (Rännar et al., 1994) para muitos preditores e muitas saídas com poucos sujeitos.

Trabalha com o kernel linear K = X Xᵀ (n × n), então o custo independe das ~60 mil variáveis de entrada.
Previsão: Ŷ = K_t U (Tᵀ K U)⁻¹ Tᵀ Y, com T/U os escores de X/Y extraídos por NIPALS sobre K deflacionado.
"""

import numpy as np


class KernelPLS:
    def __init__(self, n_components=3, max_iter=500, tol=1e-10):
        self.A = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, Y):
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-12
        self.ym = Y.mean(0)
        self.Xtr = (X - self.xm) / self.xs
        Yc = Y - self.ym
        K = self.Xtr @ self.Xtr.T
        n = len(K)
        Kd, Yd = K.copy(), Yc.copy()
        T, U = [], []
        for _ in range(min(self.A, n - 1)):
            u = Yd[:, np.argmax(Yd.var(0))].copy()
            u /= np.linalg.norm(u) + 1e-300
            t_old = None
            for _ in range(self.max_iter):
                t = Kd @ u
                t /= np.linalg.norm(t) + 1e-300
                c = Yd.T @ t
                u = Yd @ c
                u /= np.linalg.norm(u) + 1e-300
                if t_old is not None and np.linalg.norm(t - t_old) < self.tol:
                    break
                t_old = t
            T.append(t)
            U.append(u)
            P = np.eye(n) - np.outer(t, t)
            Kd = P @ Kd @ P
            Yd = Yd - np.outer(t, t @ Yd)
        T, U = np.array(T).T, np.array(U).T
        self.B = U @ np.linalg.pinv(T.T @ K @ U) @ T.T @ Yc
        return self

    def predict(self, X):
        Xt = (X - self.xm) / self.xs
        return Xt @ self.Xtr.T @ self.B + self.ym