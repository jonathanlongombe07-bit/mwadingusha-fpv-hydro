"""Consignes (standard + surcharges jusqu'à +50 %) et stress tests."""
import numpy as np


def pdem_list(cfg):
    ref = cfg.hydro.P_garantie
    std = [m * 1e6 for m in cfg.dispatch.P_std_MW]
    sur = [ref * (1 + s) for s in cfg.dispatch.surcharges]
    vals = sorted(set(round(v) for v in std + sur))
    lab = {}
    for v in vals:
        mw = v / 1e6
        if abs(v - ref) < 1: lab[v] = "60 (ref)"
        elif abs(v - cfg.hydro.P_disponible) < 1: lab[v] = "75 (P_dispo)"
        else:
            tag = [f"+{int(round(s*100))}%" for s in cfg.dispatch.surcharges if abs(ref * (1 + s) - v) < 1]
            lab[v] = f"{mw:g}" + (f" ({tag[0]})" if tag else "")
    return vals, lab


def stress_cases(cfg):
    return {"base": dict(Q_tur_max=cfg.hydro.Q_tur_max_op, Q_in_factor=1.0, nom="Base")} | \
           {k: dict(Q_tur_max=v.Q_tur_max, Q_in_factor=v.Q_in_factor, nom=v.nom) for k, v in vars(cfg.stress).items()}


def low_level_mask(V_daily, quantile=0.25):
    """Jours de bas niveau : volume sous le quantile donné de la série observée."""
    return V_daily <= np.quantile(V_daily, quantile)
