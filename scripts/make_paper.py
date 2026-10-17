"""Consolida os resultados em figuras (PNG + PDF) e numa tabela-resumo para o artigo: results/paper/.

  python scripts/make_paper.py
Partes ausentes (ainda não rodadas) são puladas com aviso.
"""

import csv
import json
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from matplotlib.ticker import PercentFormatter  # noqa: E402

from neurochaos.config import resolve  # noqa: E402

# Paleta categórica de referência (ordem fixa) + neutro para controles; tinta de texto separada da cor das séries.
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
NEUTRO, TINTA, TINTA2, GRADE = "#8f8d87", "#0b0b0b", "#52514e", "#e4e3df"
COND_COR = {"propria": C1, "real": C1, "outro_sujeito": C2, "grupo": C3, "embaralhada": NEUTRO, "B2": C4}
COND_NOME = {"propria": "própria", "real": "própria", "outro_sujeito": "outra pessoa", "grupo": "média do grupo",
             "embaralhada": "embaralhada", "B2": "prevista (B2)"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA,
    "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white", "legend.frameon": False,
})

R = resolve("results")
OUT = R / "paper"
OUT.mkdir(parents=True, exist_ok=True)
resumo = []


def load(p):
    p = R / p
    if not p.exists():
        print("  (ausente)", p)
        return None
    return json.loads(p.read_text(encoding="utf-8")) if p.suffix == ".json" else list(csv.DictReader(open(p, encoding="utf-8")))


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  figura:", name)


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    bs = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return np.percentile(bs, [2.5, 97.5])


# ------------------------------------------------------------------ Fig. 1: escada de modelos (estrutura)
def fig_escada():
    rows = []  # (nome, r_desvio, identificação, grupo)
    b = load("baselines/schaefer200/summary.json")
    if b:
        S = {s["model"]: s for s in b["summary"]}
        rows += [("Distância geodésica (B1)", S["B1_distancia"]["pearson_desvio"], S["B1_distancia"]["ident_accuracy"], "base"),
                 ("Linear FC+anatomia (B2)", S["B2_linear_multimodal"]["pearson_desvio"], S["B2_linear_multimodal"]["ident_accuracy"], "base"),
                 ("Ridge entre sujeitos (B3)", S["B3_ridge_FC"]["pearson_desvio"], S["B3_ridge_FC"]["ident_accuracy"], "base")]
    e = load("escada/pls_curva.json")
    if e:
        rows.append(("PLS de kernel", e["models"]["PLS"]["pearson_desvio"], e["models"]["PLS"]["ident_accuracy"], "ml"))
    g1 = load("gan/schaefer200/summary.json")
    if g1:
        S = {s["model"]: s for s in g1["summary"]}
        rows.append(("GAN pix2pix (v1)", S["GAN_pix2pix_ens"]["pearson_desvio"], S["GAN_pix2pix_ens"]["ident_accuracy"], "gan"))
    g2 = load("gan/schaefer200_resB2/summary.json")
    if g2:
        S = {s["model"]: s for s in g2["summary"]}
        rows.append(("GAN residual sobre B2", S["GAN_pix2pix_resB2_ens"]["pearson_desvio"], S["GAN_pix2pix_resB2_ens"]["ident_accuracy"], "gan"))
    gn = load("gan_nested/schaefer200/summary.json")
    if gn:
        rows.append(("Busca aninhada GAN/BrainNetCNN", gn["secondary"]["ajustado"]["pearson_desvio"],
                     gn["secondary"]["ajustado"]["ident_accuracy"], "gan"))
    gnn = load("escada/gnn/schaefer200/summary.json")
    if gnn:
        rows.append(("Rede de grafos (GNN)", gnn["secondary"]["ajustado"]["pearson_desvio"],
                     gnn["secondary"]["ajustado"]["ident_accuracy"], "ml"))
    ec = load("escada/ec.json")
    if ec:
        rows.append(("Conectividade efetiva (Hopf)", ec["models"]["EC"]["pearson_desvio"], ec["models"]["EC"]["ident_accuracy"], "sim"))
    if not rows:
        return
    names = [r[0] for r in rows]
    cor = {"base": NEUTRO, "ml": C1, "gan": C2, "sim": C3}
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 0.42 * len(rows) + 1.3), sharey=True)
    y = np.arange(len(rows))
    for k, (col, titulo) in enumerate(((1, "r do desvio individual"), (2, "Identificação (acurácia)"))):
        v = [r[col] for r in rows]
        ax[k].barh(y, v, height=0.62, color=[cor[r[3]] for r in rows])
        for yi, vi in zip(y, v):
            ax[k].text(vi + (0.002 if k == 0 else 0.004), yi, f"{vi:.3f}" if k == 0 else f"{vi:.0%}",
                       va="center", fontsize=7.5, color=TINTA2)
        ax[k].set_title(titulo, loc="left", color=TINTA)
        ax[k].grid(axis="y", visible=False)
    ax[1].axvline(0.02, color=TINTA2, lw=1, ls="--")
    ax[1].text(0.03, len(rows) - 0.35, "← acaso (2%)", fontsize=7.5, color=TINTA2, va="center")
    ax[1].xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax[0].set_yticks(y, names)
    ax[0].set_ylim(len(rows) - 0.1, -0.6)
    from matplotlib.patches import Patch
    fig.legend([Patch(color=cor[c]) for c in ("base", "ml", "gan", "sim")],
               ["baselines", "aprendizado (não adversarial)", "família GAN", "simulação (EC)"],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04), fontsize=8)
    fig.suptitle("Inferir a SC individual a partir de BOLD + anatomia (MICA-MICs, n = 50, Schaefer-200)",
                 x=0.01, ha="left", fontsize=10.5, color=TINTA)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig1_escada_modelos")
    resumo.append(("Escada de modelos (r do desvio / identificação)",
                   "; ".join(f"{n}: {r:.3f} / {i:.0%}" for n, r, i, _ in rows)))


# ------------------------------------------------------------------ Fig. 2: curva de aprendizado
def fig_curva():
    e = load("escada/pls_curva.json")
    if not e:
        return
    pts = e["learning_curve"]["points"]
    Ns = sorted({p["N"] for p in pts})
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for key, cor in (("B2", C1), ("PLS", C2)):
        m = [np.mean([p[key] for p in pts if p["N"] == N]) for N in Ns]
        sd = [np.std([p[key] for p in pts if p["N"] == N]) for N in Ns]
        ax.errorbar(Ns, m, yerr=sd, color=cor, lw=2, marker="o", ms=5, capsize=3, label=key)
        ax.text(Ns[-1] + 1, m[-1], key, color=TINTA, va="center", fontsize=8.5)
    ax.set_xlabel("sujeitos de treino")
    ax.set_ylabel("r do desvio individual (teste)")
    ax.set_xticks(Ns)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Curva de aprendizado: o linear satura, o PLS ainda sobe", loc="left")
    ax.text(0.98, 0.04, "identificação no teste: B2 12% · PLS 0%\n(o ganho do PLS é populacional, não individual)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=TINTA2)
    save(fig, "fig2_curva_aprendizado")
    resumo.append(("Curva de aprendizado (N=10→40)", "; ".join(
        f"N={N}: B2 {np.mean([p['B2'] for p in pts if p['N'] == N]):.3f}, PLS {np.mean([p['PLS'] for p in pts if p['N'] == N]):.3f}"
        for N in Ns)))


# ------------------------------------------------------------------ Fig. 3: simulações × BOLD por conectoma
def painel_condicoes(ax, rows, conds, col, titulo, key_cond="cond", wide=False):
    for i, c in enumerate(conds):
        if wide:
            v = np.array([float(r[f"{c}_{col}"]) for r in rows])
        else:
            v = np.array([float(r[col]) for r in rows if r[key_cond] == c])
        jit = np.random.default_rng(i).uniform(-0.18, 0.18, len(v))
        ax.scatter(np.full(len(v), i) + jit, v, s=9, color=COND_COR[c], alpha=0.35, linewidths=0)
        lo, hi = boot_ci(v)
        ax.plot([i, i], [lo, hi], color=TINTA, lw=1.6)
        ax.scatter([i], [np.nanmean(v)], s=46, color=COND_COR[c], edgecolor="white", linewidth=1.5, zorder=3)
    ax.set_xticks(range(len(conds)), [COND_NOME[c] for c in conds], rotation=20, ha="right")
    ax.set_title(titulo, loc="left", fontsize=9.5)
    ax.grid(axis="x", visible=False)


def fig_simulacoes():
    hopf = load("hopf/schaefer200/per_subject.csv")
    sp = load("cortical_subjects/K2_th0_core/per_subject.csv")
    th = load("cortical_subjects/K2_th0.1_core/per_subject.csv")
    panels = [p for p in (hopf, sp, th) if p]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.4), sharey=False)
    axes = np.atleast_1d(axes)
    k = 0
    if hopf:
        painel_condicoes(axes[k], hopf, ["real", "outro_sujeito", "grupo", "embaralhada"], "teste_B",
                         "Hopf (200 regiões)", wide=True)
        axes[k].set_ylabel("FC simulada × FC empírica (r)")
        k += 1
    conds = ["propria", "outro_sujeito", "grupo", "embaralhada"]
    if sp:
        painel_condicoes(axes[k], sp, conds, "fc_lenta_x_bold", "NEURON RS/FS, espontâneo (70 regiões)")
        k += 1
    if th:
        painel_condicoes(axes[k], th, conds, "fc_lenta_x_bold", "NEURON RS/FS, theta 6 Hz")
    fig.suptitle("Qual conectoma reproduz o BOLD da pessoa? (n = 50; ponto = média, barra = IC 95%)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig3_simulacoes_conectoma")
    for nome, rows, col, wide in (("Hopf", hopf, "teste_B", True), ("NEURON espontâneo", sp, "fc_lenta_x_bold", False),
                                  ("NEURON theta", th, "fc_lenta_x_bold", False)):
        if not rows:
            continue
        cs = ["real", "outro_sujeito", "grupo", "embaralhada"] if wide else conds
        resumo.append((f"{nome}: FC simulada × BOLD", "; ".join(
            f"{COND_NOME[c]} {np.mean([float(r[f'{c}_{col}']) if wide else float(r[col]) for r in rows if wide or r['cond'] == c]):.3f}"
            for c in cs)))


# ------------------------------------------------------------------ Fig. 4: conectividade efetiva
def fig_ec():
    ec = load("escada/ec.json")
    rows = load("escada/ec_per_subject.csv")
    if not ec or not rows:
        return
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.3))
    keys = [("fc_inicial_x_B", "SC média (início)", NEUTRO), ("fc_EC_x_B", "EC individual", C3), ("teto_A_x_B", "teto (FC A × B)", C1)]
    for i, (k, nome, cor) in enumerate(keys):
        v = np.array([float(r[k]) for r in rows])
        ax[0].scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-.18, .18, len(v)), v, s=9, color=cor,
                      alpha=.35, linewidths=0)
        lo, hi = boot_ci(v)
        ax[0].plot([i, i], [lo, hi], color=TINTA, lw=1.6)
        ax[0].scatter([i], [v.mean()], s=46, color=cor, edgecolor="white", linewidth=1.5, zorder=3)
    ax[0].set_xticks(range(3), [k[1] for k in keys], rotation=15, ha="right")
    ax[0].set_ylabel("FC do modelo × FC da 2ª metade (r)")
    ax[0].set_title("Função: a EC prevê o BOLD não visto", loc="left", fontsize=9.5)
    ax[0].grid(axis="x", visible=False)
    m = ec["models"]
    labs = ["EC individual", "B2 (mesma FC)"]
    vals = [m["EC"]["ident_accuracy"], m["B2_fcA"]["ident_accuracy"]]
    ax[1].bar(range(2), vals, width=0.55, color=[C3, C4])
    for i, v in enumerate(vals):
        ax[1].text(i, v + 0.004, f"{v:.0%}", ha="center", fontsize=8, color=TINTA2)
    ax[1].axhline(0.02, color=TINTA2, ls="--", lw=1)
    ax[1].text(1.3, 0.022, "acaso", fontsize=7.5, color=TINTA2)
    ax[1].set_xticks(range(2), labs)
    ax[1].set_ylabel("identificação pela SC da difusão")
    ax[1].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax[1].set_title("Estrutura: a EC não reconhece a SC da pessoa", loc="left", fontsize=9.5)
    ax[1].grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig4_conectividade_efetiva")
    resumo.append(("Conectividade efetiva", f"FC×2ª metade: início {ec['fc']['fc_inicial_x_B']:.3f} → EC "
                   f"{ec['fc']['fc_EC_x_B']:.3f} (teto {ec['fc']['teto_A_x_B']:.3f}); identificação EC "
                   f"{m['EC']['ident_accuracy']:.0%} vs B2 {m['B2_fcA']['ident_accuracy']:.0%}; r_desvio EC "
                   f"{m['EC']['pearson_desvio']:.3f} vs B2 {m['B2_fcA']['pearson_desvio']:.3f} (p={ec['teste_D']['p_wilcoxon']:.2g})"))


# ------------------------------------------------------------------ Fig. 5: caos na rede cortical
def fig_caos():
    c = load("escada/caos_cortical/summary.json")
    if not c:
        return
    rows = c["rows"]
    subs = c["subjects"]
    eps = sorted([k for k in rows[0] if k.startswith("p_div_eps")], key=lambda k: float(k[9:]))
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.2))
    for cond, cor, dx in (("propria", C1, -0.12), ("embaralhada", NEUTRO, 0.12)):
        m = [np.mean([r[k] for r in rows if r["cond"] == cond]) for k in eps]
        ax[0].plot(np.arange(len(eps)) + dx, m, "o-", color=cor, lw=2, ms=6, label=COND_NOME[cond])
    ax[0].set_xticks(range(len(eps)), [f"{float(k[9:]):g} nA" for k in eps])
    ax[0].set_xlabel("tamanho da perturbação")
    ax[0].set_ylabel("fração de trajetórias que divergem")
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].legend(fontsize=8)
    ax[0].set_title("Instabilidade de tamanho finito", loc="left", fontsize=9.5)
    for cond, cor, x in (("propria", C1, 0), ("embaralhada", NEUTRO, 1)):
        v = np.array([r["lambda_linear"] for r in rows if r["cond"] == cond], float)
        ax[1].scatter(np.full(len(v), x) + np.random.default_rng(x).uniform(-.15, .15, len(v)), v, s=14, color=cor, alpha=.6)
        ax[1].scatter([x], [np.nanmean(v)], s=50, color=cor, edgecolor="white", linewidth=1.5, zorder=3)
    ax[1].axhline(0, color=TINTA2, lw=1)
    ax[1].set_xticks([0, 1], ["própria", "embaralhada"])
    ax[1].set_ylabel("λ linear (1/s)")
    ax[1].set_title("Perturbações mínimas se dissipam (λ < 0)", loc="left", fontsize=9.5)
    ax[1].grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig5_caos_cortical")
    t = c["tests"]
    resumo.append(("Caos (rede cortical, 10 sujeitos)", "; ".join(
        f"{k}: própria {v['propria']:+.3f} vs embaralhada {v['embaralhada']:+.3f} (p=" + ("n/a" if v["p"] is None else f"{v['p']:.2g}") + ")" for k, v in t.items())))


def suplementares():
    pares = [("calibration_cortical_atlas/fi_curve.png", "S1_curva_FI_RS_FS.png"),
             ("validation/chaos_validation.png", "S2_validacao_caos_lorenz.png"),
             ("grid_debug/analysis/figures/metrics_vs_K.png", "S3_MVP_HH_metricas_vs_K.png"),
             ("baselines/schaefer200/baselines.png", "S4_baselines_identificacao.png")]
    for src, dst in pares:
        if (R / src).exists():
            shutil.copy(R / src, OUT / dst)
            print("  suplementar:", dst)


def main():
    print("gerando em", OUT)
    fig_escada()
    fig_curva()
    fig_simulacoes()
    fig_ec()
    fig_caos()
    suplementares()
    with open(OUT / "resumo_resultados.md", "w", encoding="utf-8") as f:
        f.write("# Resumo numérico dos resultados (gerado por scripts/make_paper.py)\n\n")
        for titulo, txt in resumo:
            f.write(f"- **{titulo}:** {txt}\n")
    print("  resumo: resumo_resultados.md")


if __name__ == "__main__":
    main()