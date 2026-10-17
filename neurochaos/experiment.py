"""Uma execução completa: rede → simulação → observáveis → métricas → arquivos (guia §8)."""

import json
import time
from itertools import combinations, product
from pathlib import Path

import numpy as np

from . import connectome as cx
from . import signals as sg
from .metrics import chaos, pac, spectral, sync
from .model import CELL_TYPES, NetworkModel, pop_name
from .jsonutil import to_jsonable  # noqa: F401  (reexportado)
from .provenance import provenance
from .stimulus import drive_phase


def _region_signals(cfg, model, res):
    s = cfg["simulation"]
    t0, t1 = float(s["transient_ms"]), float(s["duration_ms"])
    fs = float(s["signal_fs_hz"])
    sigma = float(cfg["analysis"]["rate_smoothing_ms"])
    pops = model.pop_gids()
    t = sg.time_axis(t0, t1, fs)
    rate = {}
    vmean = {}
    for lab in model.conn["labels"]:
        for ct in CELL_TYPES:
            rate[pop_name(lab, ct)] = sg.population_rate(res["spkt"], res["spkid"], pops[pop_name(lab, ct)],
                                                         t0, t1, fs, sigma)
        if "vall" in res:
            vmean[lab] = sg.mean_v(res["vall"], res["vall_gids"], pops[pop_name(lab, "E")], t0, t1,
                                   res["vall_dt_ms"])[: len(t)]
    return t, rate, vmean


def _activity_metrics(cfg, model, res):
    s = cfg["simulation"]
    t0, t1 = float(s["transient_ms"]), float(s["duration_ms"])
    out = {}
    pops = model.pop_gids()
    for ct in CELL_TYPES:
        gids = [g for p, gl in pops.items() if p.endswith("_" + ct) for g in gl]
        r = sg.cell_rates(res["spkt"], res["spkid"], gids, t0, t1)
        out[f"rate_{ct}_mean"] = float(r.mean())
        out[f"rate_{ct}_max"] = float(r.max())
        out[f"frac_silent_{ct}"] = float(np.mean(r < 0.5))
        out[f"cv_isi_{ct}"] = sg.isi_cv(res["spkt"], res["spkid"], gids, t0, t1)
    for p, gl in pops.items():
        out[f"rate_{p}"] = float(sg.cell_rates(res["spkt"], res["spkid"], gl, t0, t1).mean())
    # Forma do PA e potencial basal numa célula E e numa I com traço gravado (critérios do §7).
    for ct in CELL_TYPES:
        k = next((i for i, g in enumerate(res["trace_gids"]) if model.gid_type(g) == ct), None)
        if k is not None:
            w, vb = sg.spike_shape(res["t_trace"], res["v_trace"][k])
            out[f"ap_halfwidth_ms_{ct}"] = w
            out[f"v_p10_mV_{ct}"] = vb
    return out


def _oscillation_metrics(cfg, t, rate, model):
    an = cfg["analysis"]
    fs = float(cfg["simulation"]["signal_fs_hz"])
    edge = float(an["filter_edge_s"])
    rng = np.random.default_rng([cfg["seeds"]["stim"], 31337])
    labels = model.conn["labels"]
    sigs = {lab: rate[pop_name(lab, "E")] for lab in labels}
    out = {"psd": {}, "plv": {}, "entrain": {}, "pac": {}}

    for lab, x in sigs.items():
        out["psd"][lab], _, _ = spectral.summarize(x, fs, an["psd"]["nperseg_s"], an["bands"])

    phases = {b: {lab: np.angle(sync.analytic(x, fs, rng_b, edge)) for lab, x in sigs.items()}
              for b, rng_b in an["bands"].items()}
    nsur = int(an["plv"]["n_surrogates"])
    mshift = int(an["plv"]["min_shift_s"] * fs)
    for b in an["bands"]:
        pairs = {}
        for a, c in combinations(labels, 2):
            pairs[f"{a}|{c}"] = sync.plv_test(phases[b][a], phases[b][c], nsur, mshift, rng)
        out["plv"][b] = pairs
        # PLV de cada região com a fase do componente correspondente do estímulo.
        comp = cfg["stimulus"]["theta" if b == "theta" else "gamma"] if b in ("theta", "gamma") else None
        if comp is not None:
            e = int(round(edge * fs))
            tt = t[e: len(t) - e] if e > 0 else t
            ph_drive = drive_phase(tt, comp)
            n_cyc = (len(tt) / fs) * comp["freq_hz"]
            out["entrain"][b] = {lab: sync.entrainment_test(phases[b][lab], ph_drive, n_cyc)
                                 for lab in labels}

    pc = an["pac"]
    for lab, x in sigs.items():
        out["pac"][lab] = pac.pac_test(x, fs, pc["phase_band"], pc["amp_band"], int(pc["n_bins"]),
                                       int(pc["n_surrogates"]), float(pc["min_shift_s"]), edge, rng)
    return out


def _twin_trajectories(cfg, model, ref):
    """λ_max condicional por trajetórias gêmeas com ruído congelado."""
    ch = cfg["chaos"]
    dt = ref["vall_dt_ms"]
    H = float(ch["horizon_ms"])
    win = max(1, int(round(ch["envelope_ms"] / dt)))
    targets = sorted(model.pert)
    out = {"runs": [], "determinism_max_abs_mV": None}

    # Verificação de determinismo: repetir a referência (sem perturbação) deve reproduzir V exatamente.
    t_chk = float(min(ch["t_perturb_ms"]))
    rep = model.run(t_chk)
    n = rep["vall"].shape[1]
    out["determinism_max_abs_mV"] = float(np.max(np.abs(rep["vall"] - ref["vall"][:, :n])))

    curves_by_eps = {}
    for k, (tp, eps) in enumerate(product(ch["t_perturb_ms"], ch["eps_nA"])):
        gid = targets[k % len(targets)]
        dur = float(tp) + H
        per = model.run(dur, perturb={"gid": gid, "t_ms": tp, "amp_nA": eps, "dur_ms": ch["pulse_ms"]})
        i0 = int(round(tp / dt))
        i1 = min(per["vall"].shape[1], ref["vall"].shape[1], int(round(dur / dt)) + 1)
        pre = float(np.max(np.abs(per["vall"][:, :i0] - ref["vall"][:, :i0]))) if i0 > 0 else 0.0
        delta = chaos.divergence(ref["vall"][:, i0:i1], per["vall"][:, i0:i1])
        # δ pode voltar a ser exatamente 0 (reconvergência bit a bit): corta no piso numérico
        # para que ln(1e-300) de uma execução não domine a média do ensemble.
        L = chaos.log_envelope(delta, win, floor=float(ch["floor_mV"]))
        curves_by_eps.setdefault(float(eps), []).append(L)
        spk_ref = ref["spkt"][(ref["spkt"] >= tp) & (ref["spkt"] < dur)]
        spk_per = per["spkt"][(per["spkt"] >= tp) & (per["spkt"] < dur)]
        out["runs"].append({"t_perturb_ms": tp, "eps_nA": eps, "gid": gid,
                            "pre_perturbation_max_abs_mV": pre,
                            "delta_final_mV": float(delta[-1]),
                            "n_spikes_ref": int(len(spk_ref)), "n_spikes_pert": int(len(spk_per))})

    # Dois regimes distintos (não devem ser misturados numa média de ln δ):
    #  - linear: ε pequeno → λ (inclinação de ln δ até o piso numérico ou a saturação);
    #  - tamanho finito: fração de perturbações cujas trajetórias NÃO reconvergem
    #    (δ final > div_threshold_mV), típico de inversões de spikes ("stable chaos").
    skip = float(ch["fit_skip_ms"])
    floor = np.log(float(ch["floor_mV"]))
    thr = float(ch["div_threshold_mV"])
    finals = {}
    for r in out["runs"]:
        finals.setdefault(float(r["eps_nA"]), []).append(r["delta_final_mV"])
    per_eps = {}
    curves_out = {}
    for eps, curves in curves_by_eps.items():
        m = min(len(c) for c in curves)
        e = chaos.ensemble_lyapunov([c[:m] for c in curves], dt, skip, floor=floor + 1e-9)
        div = np.array(finals[eps]) > thr
        indiv = np.array(e["lambda_individual"]) * 1000.0
        per_eps[str(eps)] = {
            "lambda_ensemble_per_s": e["lambda"] * 1000.0, "r2": e["r2"],
            "fit_from_ms": e["fit_from"], "fit_to_ms": e["fit_to"],
            "lambda_individual_per_s": indiv.tolist(),
            "p_diverge": float(div.mean()),
            "growth_rate_diverged_per_s": float(np.median(indiv[div])) if div.any() else None,
            "lambda_reconverged_per_s": float(np.median(indiv[~div])) if (~div).any() else None,
        }
        curves_out[str(eps)] = e["mean_L"]
    eps_sorted = sorted(per_eps, key=float)
    lin = per_eps[eps_sorted[0]]
    out["per_eps"] = per_eps
    # λ linear: ε mínimo, apenas se nenhuma dessas perturbações divergiu (senão o regime não é linear).
    out["lambda_linear_per_s"] = lin["lambda_ensemble_per_s"] if lin["p_diverge"] == 0 else None
    out["p_diverge_by_eps"] = {k: per_eps[k]["p_diverge"] for k in eps_sorted}
    out["p_diverge_max_eps"] = per_eps[eps_sorted[-1]]["p_diverge"]
    out["p_diverge_all"] = float(np.mean([v > thr for vs in finals.values() for v in vs]))
    return out, curves_out


def check_windows(cfg):
    """Falha cedo, com mensagem clara, se a janela de análise não comporta filtros e surrogates."""
    s, an = cfg["simulation"], cfg["analysis"]
    usable_s = (float(s["duration_ms"]) - float(s["transient_ms"])) / 1000.0 - 2 * float(an["filter_edge_s"])
    need_s = max(2 * float(an["plv"]["min_shift_s"]), 2 * float(an["pac"]["min_shift_s"]), 1.0)
    if usable_s < need_s:
        raise ValueError(f"janela de análise útil = {usable_s:.2f} s (duração − transiente − 2×borda do filtro); "
                         f"são necessários ≥ {need_s:.2f} s. Aumente simulation.duration_ms.")
    if cfg["chaos"]["enabled"]:
        late = max(cfg["chaos"]["t_perturb_ms"]) + float(cfg["chaos"]["horizon_ms"])
        if min(cfg["chaos"]["t_perturb_ms"]) < float(s["transient_ms"]) or late > float(s["duration_ms"]):
            raise ValueError("chaos.t_perturb_ms deve ficar após o transiente e t_perturb + horizon ≤ duração.")


def run_job(cfg, out_dir, save_arrays=True):
    check_windows(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    conn = cx.build_connectome(cfg)
    cx.validate(conn)
    model = NetworkModel(cfg, conn)
    t_build = time.time() - t_start

    res = model.run(float(cfg["simulation"]["duration_ms"]))
    t_sim = time.time() - t_start - t_build

    t, rate, vmean = _region_signals(cfg, model, res)
    metrics = {"activity": _activity_metrics(cfg, model, res)}
    metrics.update(_oscillation_metrics(cfg, t, rate, model))

    # Seção estroboscópica na fase theta e máximos locais (diagramas de bifurcação).
    th = cfg["stimulus"]["theta"]
    strobe = {}
    if vmean:
        for lab, v in vmean.items():
            m = min(len(t), len(v))  # CoreNEURON pode gravar 1 amostra a menos que o eixo de tempo
            strobe[lab] = chaos.stroboscopic(t[:m], v[:m], th["freq_hz"], th["phase_rad"])[1]

    arrays = {}
    if cfg["chaos"]["enabled"]:
        metrics["chaos"], curves = _twin_trajectories(cfg, model, res)
        for eps, L in curves.items():
            arrays[f"chaos_L_eps{eps}"] = L

    meta = {
        "config": cfg,
        "provenance": provenance(cfg),
        "connectome": {k: v for k, v in conn.items()},
        "n_cells": len(model.cells),
        "n_connections": model.n_connections(),
        "bias_nA_mean": {ct: float(np.mean([b for g, b in model.bias_nA.items() if model.gid_type(g) == ct]))
                         for ct in CELL_TYPES},
        "timing_s": {"build": t_build, "main_sim": t_sim, "total": time.time() - t_start},
        "n_spikes": int(len(res["spkt"])),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable({"meta": meta, "metrics": metrics}), f, indent=1)

    if save_arrays:
        pops = model.pop_gids()
        if not cfg["simulation"].get("save_traces", True):
            res = dict(res, v_trace=res["v_trace"][:, :0], i_trace=res["i_trace"][:, :0], t_trace=res["t_trace"][:0])
        np.savez_compressed(
            out_dir / "data.npz",
            spkt=res["spkt"], spkid=res["spkid"],
            pop_names=np.array(list(pops)), pop_gids=np.array([np.array(v) for v in pops.values()], dtype=object),
            t_signal=t, **{f"rate_{k}": v for k, v in rate.items()},
            **{f"vmean_{k}": v for k, v in vmean.items()},
            **{f"strobe_{k}": v for k, v in strobe.items()},
            t_trace=res["t_trace"], trace_gids=res["trace_gids"],
            v_trace=res["v_trace"].astype(np.float32), i_trace=res["i_trace"].astype(np.float32),
            P=conn["P"], delay_ms=conn["delay_ms"], distance_mm=conn["distance_mm"],
            labels=np.array(conn["labels"]), **arrays,
        )
    return metrics, meta