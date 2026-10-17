"""Busca aninhada PRÉ-REGISTRADA (docs/preregistro_gan_nested.md): GAN/U-Net/BrainNetCNN residual sobre o B2,
com aumento de dados (FC por bootstrap em blocos do BOLD + ruído), Schaefer-200, MICA-MICs.

  python scripts/run_gan_nested.py            (busca completa: 5 externos × 4 internos × 8 configs + treino final)
  python scripts/run_gan_nested.py --smoke    (teste rápido de execução; não gera resultado válido)
"""

import argparse
import hashlib
import itertools
import json
import time

import _bootstrap  # noqa: F401
import numpy as np
import torch
from scipy.stats import binomtest, wilcoxon

from neurochaos import augment, gan, mica, sc_metrics, sc_models
from neurochaos.config import resolve
from neurochaos.jsonutil import to_jsonable

PARC = "schaefer200"
GRIDS = {
    # docs/preregistro_gan_nested.md
    "gan": [{"arch": a, "lam_adv": adv, "lam_corr": lc}
            for a, adv, lc in itertools.product(("unet", "brainnet"), (0.0, 1.0), (0.0, 10.0))],
    # docs/preregistro_adendo_modelos.md, seção B
    "gnn": [{"arch": f"gnn_h{h}_l{l}", "lam_adv": 0.0, "lam_corr": lc}
            for h, l, lc in itertools.product((32, 64), (1, 2), (0.0, 10.0))],
}
PREREG = {"gan": "docs/preregistro_gan_nested.md", "gnn": "docs/preregistro_adendo_modelos.md"}


def simplicity(c):
    """Desempate: sem adversarial, sem correlação, depois o modelo menor (BrainNetCNN; GNN com menos camadas/oculta)."""
    a = c["arch"]
    size = (0, 0) if a == "brainnet" else (1, 0) if a == "unet" else tuple(int(t[1:]) for t in a.split("_")[1:][::-1])
    return (c["lam_adv"], c["lam_corr"], size)


def tag(c):
    return f"{c['arch']}_adv{c['lam_adv']:g}_corr{c['lam_corr']:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--family", choices=["gan", "gnn"], default="gan")
    a = ap.parse_args()
    torch.set_num_threads(1)
    base_dir = "results/gan_nested" if a.family == "gan" else "results/escada/gnn"
    out = resolve(base_dir) / ("smoke" if a.smoke else PARC)
    out.mkdir(parents=True, exist_ok=True)
    prereg = resolve(PREREG[a.family])
    sha = hashlib.sha256(prereg.read_bytes()).hexdigest()
    (out / "preregistro_sha256.txt").write_text(f"{sha}  {PREREG[a.family]}\n", encoding="utf-8")
    print("pré-registro SHA-256:", sha, "| dispositivo:", gan.device(), flush=True)
    epochs = 20 if a.smoke else a.epochs
    grid = GRIDS[a.family][:2] if a.smoke else GRIDS[a.family]

    mats = mica.load_matrices(PARC)
    subs = sorted(mats)
    n = len(mats[subs[0]]["labels"])
    labels = mats[subs[0]]["labels"]
    iu = np.triu_indices(n, 1)
    for s, d in mats.items():
        d["y"] = sc_models.target(d["sc"])
    feat = {s: {"y": d["y"], "fc": sc_models.upper(d["fc"]), "gd": sc_models.upper(d["gd"]),
                "mpc": sc_models.upper(d["mpc"]), "inter": sc_models.interhemispheric(labels)} for s, d in mats.items()}
    folds = mica.read_folds()

    # Banco de FCs reamostradas por sujeito (arctanh, diagonal 0) — usado só quando o sujeito está no treino.
    rng_pool = np.random.default_rng(20261015)
    pools = {}
    for s in subs:
        ts, _ = mica.load(s, PARC, "timeseries")
        P = np.arctanh(np.clip(augment.block_bootstrap_fc(ts, a.pool, 25, rng_pool), -0.999, 0.999))
        P[:, np.arange(n), np.arange(n)] = 0.0
        pools[s] = P.astype(np.float32)
    print(f"banco de FCs: {a.pool} por sujeito", flush=True)

    def tensors(train, base_subs):
        mu = np.mean([mats[s]["y"] for s in train], axis=0)
        b2 = sc_models.LinearMultimodal().fit([feat[s] for s in train])
        base = {s: b2.predict(feat[s]) for s in base_subs}
        return mu, base

    def fit_eval(train, val, cfg, seed):
        mu, base = tensors(train, train + val)
        Xtr, Ytr, Otr = gan.build_tensors(mats, train, mu, n, base)
        Xva, Yva, Ova = gan.build_tensors(mats, val, mu, n, base)
        pool = torch.tensor(np.stack([pools[s] for s in train]))
        return gan.train_v2(Xtr, Ytr, Otr, Xva, Yva, Ova, fc_pool=pool, arch=cfg["arch"], lam_adv=cfg["lam_adv"],
                            lam_corr=cfg["lam_corr"], epochs=epochs, seed=seed)

    t0 = time.time()
    inner_scores, chosen, final_preds, b2_preds, mus, curves = {}, {}, {}, {}, {}, []
    for k in sorted(set(folds.values())):
        test = [s for s in subs if folds[s] == k]
        train_all = [s for s in subs if folds[s] != k]
        rng = np.random.default_rng(500 + k)
        inner = rng.permutation(train_all)
        inner_folds = [list(inner[i::4]) for i in range(4)]
        scores = {}
        for cfg in grid:
            sc = []
            for j, ival in enumerate(inner_folds):
                itr = [s for s in train_all if s not in ival]
                _, info = fit_eval(itr, ival, cfg, seed=0)
                sc.append(info["best_r_dev_val"])
                curves.append({"outer": k, "inner": j, "cfg": tag(cfg), "best_epoch": info["best_epoch"],
                               "history": info["history"]})
            scores[tag(cfg)] = {"mean": float(np.mean(sc)), "folds": sc}
            print(f"fold {k} | {tag(cfg):28s} escore interno = {np.mean(sc):+.4f}  ({time.time() - t0:.0f}s)", flush=True)
        best_val = max(v["mean"] for v in scores.values())
        ties = [c for c in grid if abs(scores[tag(c)]["mean"] - best_val) < 1e-12]
        cfg = sorted(ties, key=simplicity)[0]
        chosen[k] = tag(cfg)
        inner_scores[k] = scores
        print(f"fold {k}: escolhida {tag(cfg)}", flush=True)

        # Treino final: 35 treino + 5 validação (semente 1000+fold), 3 sementes, média das previsões.
        rng = np.random.default_rng(1000 + k)
        val = list(rng.choice(train_all, size=5, replace=False))
        train = [s for s in train_all if s not in val]
        mu, base = tensors(train, train + val + test)
        Xtr, Ytr, Otr = gan.build_tensors(mats, train, mu, n, base)
        Xva, Yva, Ova = gan.build_tensors(mats, val, mu, n, base)
        Xte, _, _ = gan.build_tensors(mats, test, mu, n, base)
        pool = torch.tensor(np.stack([pools[s] for s in train]))
        acc = {s: [] for s in test}
        for seed in (0, 1, 2):
            G, info = gan.train_v2(Xtr, Ytr, Otr, Xva, Yva, Ova, fc_pool=pool, arch=cfg["arch"],
                                   lam_adv=cfg["lam_adv"], lam_corr=cfg["lam_corr"], epochs=epochs, seed=seed)
            curves.append({"outer": k, "final_seed": seed, "cfg": tag(cfg), "best_epoch": info["best_epoch"],
                           "history": info["history"]})
            for s, dm in zip(test, gan.predict(G, Xte)):
                acc[s].append(base[s] + dm[iu])
        for s in test:
            final_preds[s] = np.mean(acc[s], axis=0)
            b2_preds[s] = base[s]
            mus[s] = mu

    truth = {s: mats[s]["y"] for s in subs}
    rows = {"ajustado": {}, "B2": {}}
    for name, P in (("ajustado", final_preds), ("B2", b2_preds)):
        for s in subs:
            rows[name][s] = sc_metrics.edge_metrics(P[s], truth[s], mus[s], n)
    x = np.array([rows["ajustado"][s]["pearson_desvio"] for s in subs])
    y = np.array([rows["B2"][s]["pearson_desvio"] for s in subs])
    primary = {"metric": "pearson_desvio", "mean_ajustado": float(x.mean()), "mean_B2": float(y.mean()),
               "n_ajustado_maior": int(np.sum(x > y)),
               "p_wilcoxon": float(wilcoxon(x, y).pvalue) if not np.allclose(x, y) else 1.0}
    secondary = {}
    for name, P in (("ajustado", final_preds), ("B2", b2_preds)):
        idn = sc_metrics.identification(P, truth, mus)
        secondary[name] = {mt: float(np.mean([rows[name][s][mt] for s in subs]))
                           for mt in ("pearson", "pearson_desvio", "auc_top20", "pearson_forca_nodal")}
        secondary[name]["ident_accuracy"] = idn["accuracy"]
        secondary[name]["ident_p_binomial"] = float(binomtest(int(round(idn["accuracy"] * len(subs))), len(subs),
                                                              1 / len(subs), alternative="greater").pvalue)
    res = {"preregistro_sha256": sha, "smoke": a.smoke, "epochs": epochs, "pool": a.pool,
           "device": str(gan.device()), "chosen_per_fold": chosen, "inner_scores": inner_scores,
           "primary": primary, "secondary": secondary, "elapsed_s": time.time() - t0}
    (out / "summary.json").write_text(json.dumps(to_jsonable(res), indent=1), encoding="utf-8")
    (out / "curves.json").write_text(json.dumps(to_jsonable(curves)), encoding="utf-8")
    print("\nconfiguração escolhida por fold:", chosen)
    print(f"PRIMÁRIO r_desvio: ajustado {primary['mean_ajustado']:.4f} vs B2 {primary['mean_B2']:.4f} | "
          f"{primary['n_ajustado_maior']}/50 sujeitos | Wilcoxon p = {primary['p_wilcoxon']:.3g}")
    for name, v in secondary.items():
        print(f"  {name:9s} r={v['pearson']:.3f} r_desvio={v['pearson_desvio']:.4f} AUC={v['auc_top20']:.3f} "
              f"ident={v['ident_accuracy']:.2f} (p={v['ident_p_binomial']:.2g})")
    print(f"tempo: {time.time() - t0:.0f}s; saídas em {out}")


if __name__ == "__main__":
    main()