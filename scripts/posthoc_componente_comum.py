import sys; sys.path.insert(0, ".")  # executar da raiz do projeto; análise POST HOC (não pré-registrada)
import numpy as np, json
from neurochaos import mica, sc_models
from neurochaos.pls import KernelPLS
mats = mica.load_matrices("schaefer200"); subs = sorted(mats); folds = mica.read_folds()
Y = {s: sc_models.target(d["sc"]) for s, d in mats.items()}
X = {s: np.concatenate([np.arctanh(np.clip(sc_models.upper(d["fc"]), -.999, .999)), sc_models.upper(d["gd"]), sc_models.upper(d["mpc"])]) for s, d in mats.items()}
shared, pls_r, pls_r_resid, pair_pred, pair_true, dens = [], [], [], [], [], []
for k in range(5):
    te = [s for s in subs if folds[s] == k]; tr = [s for s in subs if folds[s] != k]
    mu = np.mean([Y[s] for s in tr], 0)
    D = {s: Y[s] - mu for s in te}
    P = dict(zip(te, KernelPLS(10).fit(np.array([X[s] for s in tr]), np.array([Y[s] for s in tr])).predict(np.array([X[s] for s in te]))))
    Pd = {s: P[s] - mu for s in te}
    for s in te:
        others = np.mean([D[o] for o in te if o != s], 0)       # componente comum, sem o próprio sujeito
        shared.append(np.corrcoef(others, D[s])[0, 1])
        pls_r.append(np.corrcoef(Pd[s], D[s])[0, 1])
        # PLS depois de remover dele a direção comum (projeção sobre "others")
        o = others / np.linalg.norm(others); pr = Pd[s] - (Pd[s] @ o) * o; tr_ = D[s] - (D[s] @ o) * o
        pls_r_resid.append(np.corrcoef(pr, tr_)[0, 1])
        dens.append(float(np.mean(mats[s]["sc"][np.triu_indices(200, 1)] > 0)))
    pp = np.array([Pd[s] for s in te]); C = np.corrcoef(pp); pair_pred.append(C[np.triu_indices(len(te), 1)].mean())
    tt = np.array([D[s] for s in te]); C = np.corrcoef(tt); pair_true.append(C[np.triu_indices(len(te), 1)].mean())
print("r(desvio do sujeito, média dos desvios dos OUTROS de teste) = %.4f" % np.mean(shared))
print("r_desvio PLS = %.4f ; PLS após remover a direção comum = %.4f" % (np.mean(pls_r), np.mean(pls_r_resid)))
print("correlação média ENTRE previsões PLS de sujeitos diferentes = %.3f (verdadeiros: %.3f)" % (np.mean(pair_pred), np.mean(pair_true)))
print("corr(r_desvio PLS, densidade da SC) = %.3f" % np.corrcoef(pls_r, dens)[0, 1])
json.dump({"shared_r": float(np.mean(shared)), "pls_r": float(np.mean(pls_r)), "pls_r_sem_comum": float(np.mean(pls_r_resid)),
           "corr_entre_previsoes": float(np.mean(pair_pred)), "corr_entre_verdades": float(np.mean(pair_true))},
          open("results/escada/posthoc_componente_comum.json", "w"), indent=1)
