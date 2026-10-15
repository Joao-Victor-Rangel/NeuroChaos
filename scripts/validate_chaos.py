"""Validação dos estimadores de caos ANTES de aplicá-los à rede (guia §8.3).

- Trajetórias gêmeas: Lorenz (λ esperado ≈ 0,906) e van der Pol (ciclo-limite, λ ≈ 0).
- Rosenstein: Lorenz, seno (periódico) e ruído branco (controle negativo: o método dá
  λ > 0 espúrio para ruído — divergência positiva sozinha não prova caos).
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from neurochaos.jsonutil import to_jsonable
from neurochaos.metrics import chaos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/validation")
    a = ap.parse_args()
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    rep = {}

    lz = chaos.twin_lorenz(n_pairs=20, seed=0)
    vdp = chaos.twin_van_der_pol(n_pairs=10, seed=0)
    rep["twin_lorenz"] = {"lambda": lz["lambda"], "r2": lz["r2"], "expected": chaos.LORENZ_LAMBDA,
                          "fit": [lz["fit_from"], lz["fit_to"]]}
    rep["twin_van_der_pol"] = {"lambda": vdp["lambda"], "r2": vdp["r2"], "expected": 0.0}

    dt = 0.01
    x = chaos.integrate_rk4(chaos.lorenz_rhs, [1.0, 1.0, 1.0], dt, 25000)[5000:, 0]
    t = np.arange(len(x)) * dt
    sine = np.sin(2 * np.pi * t / 0.75)
    noise = np.random.default_rng(1).standard_normal(len(x))
    kw = dict(dt=dt, m=5, tau=10, theiler=100, k_max=400, fit_range=(0.5, 1.5), n_ref=1000)
    ros = {}
    for name, sig in (("lorenz", x), ("sine", sine), ("white_noise", noise)):
        lam, k, curve = chaos.rosenstein(sig, **kw)
        ros[name] = {"lambda": lam, "k": k, "curve": curve}
    rep["rosenstein"] = {k: v["lambda"] for k, v in ros.items()}
    # Mesma série de ruído, janela de ajuste curta (0–0,5): o "λ" vira fortemente positivo.
    k, c = ros["white_noise"]["k"], ros["white_noise"]["curve"]
    short = (k >= 0.0) & (k <= 0.5)
    rep["rosenstein_noise_short_window"] = float(np.polyfit(k[short], c[short], 1)[0])
    rep["rosenstein_params"] = {k: v for k, v in kw.items()}

    checks = {
        "twin_lorenz_within_15pct": abs(lz["lambda"] - chaos.LORENZ_LAMBDA) / chaos.LORENZ_LAMBDA < 0.15,
        "twin_vdp_near_zero": abs(vdp["lambda"]) < 0.1,
        "rosenstein_lorenz_within_25pct": abs(ros["lorenz"]["lambda"] - chaos.LORENZ_LAMBDA) / chaos.LORENZ_LAMBDA < 0.25,
        "rosenstein_sine_near_zero": abs(ros["sine"]["lambda"]) < 0.15,
        "rosenstein_noise_spurious_in_short_window": rep["rosenstein_noise_short_window"] > 0.3,
    }
    rep["checks"] = checks
    (od / "chaos_validation.json").write_text(json.dumps(to_jsonable(rep), indent=1), encoding="utf-8")
    for k, v in rep.items():
        if k not in ("rosenstein_params",):
            print(k, v)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        for name, e in (("Lorenz", lz), ("van der Pol", vdp)):
            ax[0].plot(e["tau"], e["mean_L"], label=f"{name}: λ={e['lambda']:.3f}")
        tt = lz["tau"]
        ax[0].plot(tt, lz["mean_L"][0] + chaos.LORENZ_LAMBDA * tt, "k--", lw=0.8, label="inclinação 0,906")
        ax[0].set_xlabel("τ (unid. de tempo)")
        ax[0].set_ylabel("⟨ln δ⟩")
        ax[0].set_title("Trajetórias gêmeas")
        ax[0].legend(fontsize=8)
        for name, r in ros.items():
            ax[1].plot(r["k"], r["curve"] - r["curve"][0], label=f"{name}: λ={r['lambda']:.2f}")
        ax[1].axvspan(*kw["fit_range"], color="0.9")
        ax[1].set_xlabel("k·dt")
        ax[1].set_ylabel("⟨ln d(k)⟩ − ⟨ln d(0)⟩")
        ax[1].set_title("Rosenstein")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(od / "chaos_validation.png", dpi=130)
    except Exception as e:
        print("figura não gerada:", e)
    if not all(checks.values()):
        raise SystemExit(f"VALIDAÇÃO FALHOU: {checks}")


if __name__ == "__main__":
    main()