"""Regras de célula do NetPyNE a partir da configuração (guia §7, estágios MVP e intermediário).

Formato de `cfg["cells"][tipo]`:
  diam_um, L_um, Ra, cm, mechs: {mecanismo: {parâmetro: valor}}, ions: {na: E_Na, k: E_K} (opcional)
Compatível com o formato antigo do MVP (`hh: {...}` em vez de `mechs`).

Mecanismos corticais (mod/, compilar com nrnivmodl): na_tm, kd_tm (Traub–Miles) e im_pos (corrente M),
na forma de Pospischil et al. (2008). Os valores de condutância são pontos de partida a calibrar.
"""


def cell_rule(c):
    mechs = dict(c.get("mechs") or {"hh": dict(c["hh"])})
    sec = {
        "geom": {"diam": c["diam_um"], "L": c["L_um"], "Ra": c["Ra"], "cm": c["cm"]},
        "mechs": {k: dict(v) for k, v in mechs.items()},
    }
    if c.get("ions"):
        sec["ions"] = {ion: {"e": float(e)} for ion, e in c["ions"].items()}
    return {"secs": {"soma": sec}}