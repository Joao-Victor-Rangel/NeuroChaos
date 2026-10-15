"""Figura-resumo de UMA execução: raster, taxas, traços + corrente entregue, PSD, caos, mapas.

  python scripts/plot_run.py results/single/exemplo
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from neurochaos.metrics import spectral


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    a = ap.parse_args()
    d = Path(a.run_dir)
    z = np.load(d / "data.npz", allow_pickle=True)
    meta = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    cfg = meta["meta"]["config"]
    labels = list(z["labels"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 3)
    t0 = float(cfg["simulation"]["transient_ms"])
    t_show = (t0, t0 + 1000.0)

    ax = fig.add_subplot(gs[0, :2])
    pops = list(z["pop_names"])
    for p, gids in zip(pops, z["pop_gids"]):
        m = np.isin(z["spkid"], gids) & (z["spkt"] >= t_show[0]) & (z["spkt"] < t_show[1])
        ax.plot(z["spkt"][m], z["spkid"][m], "|", ms=3, color="tab:red" if p.endswith("_I") else "tab:blue")
    ax.set_xlim(*t_show)
    ax.set_ylabel("gid")
    ax.set_title("Raster (azul E, vermelho I) — 1 s após o transiente")

    ax = fig.add_subplot(gs[1, :2])
    t = z["t_signal"]
    for lab in labels:
        ax.plot(t, z[f"rate_{lab}_E"], lw=0.7, label=lab)
    ax.set_xlim(*t_show)
    ax.set_ylabel("taxa E (Hz/célula)")
    ax.legend(fontsize=7, ncol=4)

    ax = fig.add_subplot(gs[2, :2])
    if z["t_trace"].size:
        tt = z["t_trace"]
        m = (tt >= t_show[0]) & (tt < t_show[1])
        ax.plot(tt[m], z["v_trace"][0][m], lw=0.6, color="k")
        ax.set_ylabel(f"V gid {int(z['trace_gids'][0])} (mV)")
        ax2 = ax.twinx()
        ax2.plot(tt[m], z["i_trace"][0][m], lw=0.6, color="tab:orange")
        ax2.set_ylabel("I entregue (nA)", color="tab:orange")
    ax.set_xlabel("t (ms)")

    ax = fig.add_subplot(gs[0, 2])
    fs = float(cfg["simulation"]["signal_fs_hz"])
    for lab in labels:
        f, p, _ = spectral.psd(z[f"rate_{lab}_E"], fs, cfg["analysis"]["psd"]["nperseg_s"])
        ax.semilogy(f[f >= 1], p[f >= 1], lw=0.8, label=lab)
    for b in cfg["analysis"]["bands"].values():
        ax.axvspan(*b, color="0.9")
    ax.set_xlim(0, 120)
    ax.set_xlabel("Hz")
    ax.set_title("PSD (Welch) da taxa E")

    ax = fig.add_subplot(gs[1, 2])
    keys = sorted([k for k in z.files if k.startswith("chaos_L_eps")], key=lambda k: float(k[11:]))
    for k in keys:
        L = z[k]
        ax.plot(np.arange(len(L)) * 1000.0 / fs, L / np.log(10), lw=0.8, label=f"ε={float(k[11:]):g} nA")
    ax.set_xlabel("τ após perturbação (ms)")
    ax.set_ylabel("⟨log10 δ⟩ (mV)")
    ch = meta["metrics"].get("chaos", {})
    ax.set_title(f"Gêmeas: λ_lin={ch.get('lambda_linear_per_s') or float('nan'):.1f}/s, "
                 f"p_div={ch.get('p_diverge_all', float('nan')):.2f}", fontsize=9)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[2, 2])
    k0 = f"strobe_{labels[0]}"
    if k0 in z and len(z[k0]) > 2:
        s = z[k0]
        ax.plot(s[:-1], s[1:], ".", ms=3)
        ax.set_xlabel("V̄ₖ (mV)")
        ax.set_ylabel("V̄ₖ₊₁ (mV)")
        ax.set_title(f"Mapa de retorno estroboscópico ({labels[0]})", fontsize=9)

    st = cfg["stimulus"]
    fig.suptitle(f"{d.name}: topo={cfg['connectome']['topology']}, K={cfg['coupling']['K']}, "
                 f"Aθ={st['theta']['amp_nA']} nA, Aγ={st['gamma']['amp_nA']} nA", fontsize=11)
    fig.tight_layout()
    fig.savefig(d / "summary.png", dpi=120)
    print("salvo:", d / "summary.png")


if __name__ == "__main__":
    main()