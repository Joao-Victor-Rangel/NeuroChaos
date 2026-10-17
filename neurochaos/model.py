"""Construção e execução da rede no NetPyNE/NEURON (guia §7).

Rede MVP: N regiões × (E, I) células HH de um compartimento, sinapses Exp2Syn,
conexões locais por probabilidade e conexões longas E→(E,I) definidas pelo conectoma.
Entrada: IClamp por célula com I(t) tocado via Vector.play + (opcional) Poisson congelado.
"""

import numpy as np
from netpyne import sim, specs
from netpyne.sim.run import preRun
from neuron import h

from .cells import cell_rule
from .stimulus import is_periodic, periodic_current

CELL_TYPES = ("E", "I")


def pop_name(label, ctype):
    return f"{label}_{ctype}"


def build_net_params(cfg, conn):
    net = specs.NetParams()
    net.defaultThreshold = cfg["simulation"].get("spike_threshold_mv", 0.0)

    for ct in CELL_TYPES:
        net.cellParams[ct] = cell_rule(cfg["cells"][ct])

    npop = cfg["populations"]
    for lab in conn["labels"]:
        net.popParams[pop_name(lab, "E")] = {"cellType": "E", "numCells": npop["n_E_per_region"]}
        net.popParams[pop_name(lab, "I")] = {"cellType": "I", "numCells": npop["n_I_per_region"]}

    for name, s in cfg["synapses"].items():
        net.synMechParams[name] = {"mod": "Exp2Syn", "tau1": s["tau1"], "tau2": s["tau2"], "e": s["e"]}

    # Conexões locais (dentro de cada região): hipótese de rede, não obtida de imagem.
    loc = cfg["local_connectivity"]
    for lab in conn["labels"]:
        for pair in ("EE", "EI", "IE", "II"):
            pre, post = pair[0], pair[1]
            p = loc[pair]
            if p["prob"] <= 0 or p["weight_uS"] <= 0:
                continue
            net.connParams[f"{lab}:{pair}"] = {
                "preConds": {"pop": pop_name(lab, pre)},
                "postConds": {"pop": pop_name(lab, post)},
                "probability": float(p["prob"]),
                "weight": float(p["weight_uS"]),
                "delay": float(loc["delay_ms"]),
                "synMech": "AMPA" if pre == "E" else "GABA",
            }

    # Conexões inter-regionais E_i -> E_j e E_i -> I_j, escaladas por K·K_base.
    cp = cfg["coupling"]
    w_long = float(cp["K"]) * float(cp["K_base_uS"])
    P, D = conn["P"], conn["delay_ms"]
    for i, src in enumerate(conn["labels"]):
        for j, tgt in enumerate(conn["labels"]):
            if i == j or P[i, j] <= 0 or w_long <= 0:
                continue
            for post, ratio in (("E", 1.0), ("I", float(cp["EI_ratio"]))):
                if ratio <= 0:
                    continue
                net.connParams[f"{src}->{tgt}:E{post}"] = {
                    "preConds": {"pop": pop_name(src, "E")},
                    "postConds": {"pop": pop_name(tgt, post)},
                    "probability": float(P[i, j]),
                    "weight": w_long * ratio,
                    "delay": float(D[i, j]),
                    "synMech": "AMPA",
                }

    # Fundo Poisson (ruído congelado: mesma realização para a mesma semente de estímulo).
    bk = cfg["background"]
    if bk["mode"] == "poisson":
        net.stimSourceParams["bkg"] = {"type": "NetStim", "rate": float(bk["rate_hz"]), "noise": 1.0,
                                       "start": 0.0, "number": 1e9}
        for ct in CELL_TYPES:
            w = float(bk["weight_uS"][ct])
            if w > 0:
                net.stimTargetParams[f"bkg->{ct}"] = {"source": "bkg", "conds": {"cellType": ct},
                                                      "weight": w, "delay": 1.0, "synMech": "AMPA"}

    # IClamp de entrada; a amplitude é definida/tocada depois de criar as células.
    net.stimSourceParams["drive"] = {"type": "IClamp", "del": 0.0, "dur": 1e9, "amp": 0.0}
    for ct in CELL_TYPES:
        net.stimTargetParams[f"drive->{ct}"] = {"source": "drive", "conds": {"cellType": ct},
                                                "sec": "soma", "loc": 0.5}
    return net


def build_sim_config(cfg):
    s = cfg["simulation"]
    sc = specs.SimConfig()
    sc.duration = float(s["duration_ms"])
    sc.dt = float(s["dt_ms"])
    sc.cvode_active = False
    sc.hParams = {"celsius": float(s["celsius"]), "v_init": float(s["v_init_mv"])}
    sc.seeds = {"conn": int(cfg["seeds"]["conn"]), "stim": int(cfg["seeds"]["stim"]),
                "loc": int(cfg["seeds"]["loc"])}
    sc.recordTraces = {}
    sc.recordStep = float(s["record_step_ms"])
    sc.recordCellsSpikes = -1
    sc.verbose = False
    sc.printPopAvgRates = False
    sc.printRunTime = False
    sc.saveJson = False
    sc.savePickle = False
    sc.analysis = {}
    # CoreNEURON (motor otimizado do NEURON, CPU): exige Random123 para o ruído dos NetStims e
    # mecanismos compilados com `nrnivmodl -coreneuron`. random123 também pode ser ligado sozinho,
    # para comparar NEURON e CoreNEURON com a MESMA realização de ruído.
    if s.get("coreneuron", False) or s.get("random123", False):
        sc.random123 = True
    if s.get("coreneuron", False):
        sc.coreneuron = True
        sc.cache_efficient = True
    return sc


class NetworkModel:
    """Rede criada uma vez; `run()` pode ser chamado várias vezes (trajetórias gêmeas)."""

    def __init__(self, cfg, conn):
        self.cfg = cfg
        self.conn = conn
        self.net_params = build_net_params(cfg, conn)
        self.sim_config = build_sim_config(cfg)
        sim.create(netParams=self.net_params, simConfig=self.sim_config, output=False)
        self.cells = sorted(sim.net.cells, key=lambda c: c.gid)
        self._by_gid = {c.gid: c for c in self.cells}
        self._hold = []  # mantém referências a Vectors/objetos NEURON vivos
        self._attach_drive()
        self._attach_perturbation_clamps()
        self._setup_recorders()

    # ------------------------------------------------------------------ entrada
    def _attach_drive(self):
        cfg = self.cfg
        st, bk = cfg["stimulus"], cfg["background"]
        rng = np.random.default_rng([cfg["seeds"]["stim"], 104729])
        dur = float(cfg["simulation"]["duration_ms"]) + 10.0
        wdt = float(st["waveform_dt_ms"])
        tvec_np = np.arange(0.0, dur + wdt, wdt)
        self.drive_t = h.Vector(tvec_np)
        self.bias_nA = {}
        self.drive_obj = {}
        periodic = is_periodic(st)
        for c in self.cells:
            ct = c.tags["cellType"]
            bias = float(bk["rheobase_nA"][ct]) * float(bk["bias_frac_rheobase"][ct])
            bias *= 1.0 + float(bk["bias_cv"]) * rng.standard_normal()
            self.bias_nA[c.gid] = bias
            stim = next(s for s in c.stims if s.get("source") == "drive")
            obj = stim["hObj"]
            self.drive_obj[c.gid] = obj
            if periodic and ct in st["targets"]:
                vec = h.Vector(periodic_current(tvec_np, bias, st["theta"], st["gamma"]))
                vec.play(obj._ref_amp, self.drive_t, True)
                self._hold.append(vec)
            else:
                obj.amp = bias

    def _attach_perturbation_clamps(self):
        """Um IClamp (amp=0) por célula-alvo candidata, presente em TODAS as execuções.

        A estrutura do modelo é idêntica entre referência e perturbada; só muda `amp`.
        """
        ch = self.cfg["chaos"]
        self.pert = {}
        for lab in self.conn["labels"]:
            gids = self.pop_gids()[pop_name(lab, "E")]
            for gid in gids[: int(ch.get("targets_per_region", 1))]:
                ic = h.IClamp(self._soma(gid)(0.5))
                ic.amp, ic.dur, ic.delay = 0.0, 0.0, 0.0
                self.pert[gid] = ic

    # ------------------------------------------------------------------ gravação
    def _soma(self, gid):
        return self._by_gid[gid].secs["soma"]["hObj"]

    def _setup_recorders(self):
        s = self.cfg["simulation"]
        rstep = float(s["record_step_ms"])
        self.t_trace = h.Vector().record(h._ref_t, rstep)
        self.trace_gids = []
        pops = self.pop_gids()
        for lab in self.conn["labels"]:
            for ct in CELL_TYPES:
                self.trace_gids.extend(pops[pop_name(lab, ct)][: int(s["n_trace_cells_per_pop"])])
        self.v_trace = {g: h.Vector().record(self._soma(g)(0.5)._ref_v, rstep) for g in self.trace_gids}
        # Corrente efetivamente entregue pelo IClamp (guia §7: inspecionar a corrente real).
        self.i_trace = {g: h.Vector().record(self.drive_obj[g]._ref_i, rstep) for g in self.trace_gids}
        # V de todas as células na resolução do sinal agregado (1 kHz por padrão).
        self.vall_dt = 1000.0 / float(s["signal_fs_hz"])
        self.v_all = {}
        if s.get("record_all_v", True):
            self.v_all = {c.gid: h.Vector().record(self._soma(c.gid)(0.5)._ref_v, self.vall_dt)
                          for c in self.cells}

    # ------------------------------------------------------------------ execução
    def run(self, duration_ms, perturb=None):
        """Integra de t=0 até duration_ms. `perturb` = dict(gid, t_ms, amp_nA, dur_ms) ou None."""
        for ic in self.pert.values():
            ic.amp, ic.dur, ic.delay = 0.0, 0.0, 0.0
        if perturb is not None:
            ic = self.pert[perturb["gid"]]
            ic.delay = float(perturb["t_ms"])
            ic.dur = float(perturb["dur_ms"])
            ic.amp = float(perturb["amp_nA"])
        sim.cfg.duration = float(duration_ms)
        sim.simData["spkt"].resize(0)
        sim.simData["spkid"].resize(0)
        preRun()  # reinicia os geradores do NetStim: mesma realização de ruído em toda execução
        h.finitialize(float(sim.cfg.hParams["v_init"]))
        if self.cfg["simulation"].get("coreneuron", False):
            from neuron import coreneuron
            coreneuron.enable = True
            coreneuron.verbose = 0
        sim.pc.psolve(float(duration_ms))
        return self._collect()

    def _collect(self):
        out = {
            "spkt": np.array(sim.simData["spkt"]),
            "spkid": np.array(sim.simData["spkid"]).astype(int),
            "t_trace": np.array(self.t_trace),
            "trace_gids": np.array(self.trace_gids),
            "v_trace": np.array([np.array(self.v_trace[g]) for g in self.trace_gids]),
            "i_trace": np.array([np.array(self.i_trace[g]) for g in self.trace_gids]),
        }
        if self.v_all:
            gids = sorted(self.v_all)
            if len(self.v_all[gids[0]]) == 0:
                raise RuntimeError("V de todas as células veio vazio: o CoreNEURON não grava Vector.record com Dt "
                                   "próprio. Use simulation.coreneuron=false quando record_all_v/chaos estiverem ligados.")
            out["vall_gids"] = np.array(gids)
            out["vall"] = np.array([np.array(self.v_all[g]) for g in gids], dtype=np.float64)
            out["vall_dt_ms"] = self.vall_dt
        return out

    def gid_type(self, gid):
        return self._by_gid[int(gid)].tags["cellType"]

    def pop_gids(self):
        pops = {}
        for c in self.cells:
            pops.setdefault(c.tags["pop"], []).append(c.gid)
        return pops

    def n_connections(self):
        return int(sum(len(c.conns) for c in self.cells))