"""Programmation dynamique JOURNALIÈRE (état = volume) + réalisation HORAIRE sous contraintes.

Principe
- étape = jour ; décision = débit moyen journalier Q_d ; état = volume V (grille).
- coût d'étape C_d(V, Q_d) = déficit minimal de la journée avec un budget d'eau Q_d*24 : l'eau est affectée
  aux heures de plus fort déficit (remplissage glouton), avec hydro NON LINÉAIRE à la charge du niveau V.
- prévision PARFAITE des apports et du PV : c'est une BORNE de référence, pas une stratégie implémentable.
- la réalisation horaire (rampe, Q_min, V_min, plafond P_dispo) est rejouée par run_rbc(q_plan=...) ;
  l'écart avec le coût prédit par la DP mesure le coût des contraintes horaires ignorées par la DP journalière.
"""
import numpy as np

from .rbc import run_rbc


def _caps(plant, z, Qmax):
    """Débit max et puissance max effectifs à la charge z (stress : Q_tur_max réduit => P max réduite)."""
    Qm = plant.Q_max if Qmax is None else min(Qmax, plant.Q_max)
    return Qm, min(plant.P_cap, float(plant.power(Qm, z)))


def _need_flows(def_h, z, plant, Qmax=None):
    """Débit nécessaire pour couvrir chaque heure de déficit (>= Q_min, <= Q_max effectif), 0 sinon."""
    Qm, Pc = _caps(plant, z, Qmax)
    qn = np.zeros(len(def_h))
    for h, d in enumerate(def_h):
        if d > 0:
            qn[h] = min(max(float(plant.q_for_power(min(d, Pc), z)), plant.Q_min), Qm)
    return qn


def hourly_plan(def_h, z, B, plant, Qmax=None):
    """Affectation gloutonne du budget B (m3/s*h) : heures de plus fort déficit d'abord. Retourne q horaires."""
    Qm, _ = _caps(plant, z, Qmax)
    qn = _need_flows(def_h, z, plant, Qmax); order = np.argsort(-def_h, kind="stable")
    q = np.zeros(len(def_h)); rem = B
    for h in order:
        if def_h[h] <= 0 or rem < plant.Q_min: continue
        take = min(qn[h], rem, Qm)
        if take >= plant.Q_min:
            q[h] = take; rem -= take
    return q


def _cost_curve(def_h, z, Qgrid, plant, Qmax=None):
    """Déficit (Wh) de la journée pour chaque débit moyen de Qgrid, à la charge z."""
    d = np.maximum(def_h, 0.0); tot = d.sum(); _, Pc = _caps(plant, z, Qmax)
    qn = _need_flows(d, z, plant, Qmax); order = np.argsort(-d, kind="stable")
    qn_s, d_s = qn[order], d[order]
    pos = d_s > 0; qn_s, d_s = qn_s[pos], d_s[pos]
    cw, cd = np.cumsum(qn_s), np.cumsum(np.minimum(d_s, Pc))   # énergie réellement fournie (plafond P_dispo)
    B = Qgrid * 24.0
    k = np.searchsorted(cw, B + 1e-9, side="right")          # heures entièrement couvertes
    covered = np.where(k > 0, cd[np.maximum(k - 1, 0)], 0.0)
    used = np.where(k > 0, cw[np.maximum(k - 1, 0)], 0.0)
    rem = B - used
    part = np.zeros_like(B)
    for j in range(len(B)):
        if k[j] < len(d_s) and rem[j] >= plant.Q_min:
            part[j] = min(float(plant.power(rem[j], z)), d_s[k[j]], Pc)
    return tot - covered - part


def solve_dp(P_fpv, P_dem, Q_in_d, V0, res, plant, cfg, Q_tur_max=None, nV=260, nQ=46, nz=7,
             pen_vmin_per_Mm3=100e9, V_end_min=None):
    n_days = len(Q_in_d); Qmax = plant.Q_max if Q_tur_max is None else Q_tur_max
    Qgrid = np.concatenate([[0.0], np.linspace(plant.Q_min, Qmax, nQ - 1)])
    V_lo = res.V_min - 5e6
    V_hi = min(res.V_max, V0 + 86400.0 * float(np.sum(np.maximum(Q_in_d, 0.0))) + 5e6)   # volume atteignable (grille DP resserrée)
    Vg = np.linspace(V_lo, V_hi, nV); zg = res.level(Vg)
    zl = np.linspace(zg.min(), zg.max(), nz)
    dt = 86400.0
    # tables de coût par jour : (n_days, nz, nQ), en Wh
    deficit = np.maximum(np.asarray(P_dem) - np.asarray(P_fpv), 0.0).reshape(n_days, 24)
    C = np.empty((n_days, nz, nQ))
    for d in range(n_days):
        for k, z in enumerate(zl):
            C[d, k] = _cost_curve(deficit[d], z, Qgrid, plant, Qmax)
    wz = np.clip((zg - zl[0]) / (zl[-1] - zl[0]) * (nz - 1), 0, nz - 1 - 1e-9)
    iz = wz.astype(int); fz = wz - iz

    def stage(d):
        return (1 - fz)[:, None] * C[d, iz] + fz[:, None] * C[d, iz + 1]     # (nV, nQ)

    J = np.zeros((n_days + 1, nV))
    if V_end_min is not None:
        J[n_days] = np.where(Vg < V_end_min, pen_vmin_per_Mm3 * (V_end_min - Vg) / 1e6, 0.0)
    for d in range(n_days - 1, -1, -1):
        Vn = Vg[:, None] + dt * (Q_in_d[d] - Qgrid[None, :])
        Vn = np.minimum(Vn, V_hi)
        pen = np.where(Vn < res.V_min, pen_vmin_per_Mm3 * (res.V_min - Vn) / 1e6, 0.0)
        feas = Vn >= V_lo
        Jn = np.interp(np.maximum(Vn, V_lo), Vg, J[d + 1])
        tot = stage(d) + pen + Jn
        tot = np.where(feas, tot, np.inf)
        J[d] = tot.min(axis=1)
    # passe avant avec le volume exact
    Vd = np.empty(n_days + 1); Vd[0] = V0; Qd = np.empty(n_days); cost = 0.0
    for d in range(n_days):
        z = float(res.level(Vd[d]))
        wzs = np.clip((z - zl[0]) / (zl[-1] - zl[0]) * (nz - 1), 0, nz - 1 - 1e-9); i0 = int(wzs); f0 = wzs - i0
        c = (1 - f0) * C[d, i0] + f0 * C[d, i0 + 1]
        Vn = np.minimum(Vd[d] + dt * (Q_in_d[d] - Qgrid), V_hi)
        pen = np.where(Vn < res.V_min, pen_vmin_per_Mm3 * (res.V_min - Vn) / 1e6, 0.0)
        tot = c + pen + np.interp(np.maximum(Vn, V_lo), Vg, J[d + 1]); tot = np.where(Vn >= V_lo, tot, np.inf)
        j = int(np.argmin(tot)); Qd[d] = Qgrid[j]; Vd[d + 1] = Vn[j]; cost += c[j]
    return dict(Q_d=Qd, V_d=Vd, cost_pred_GWh=cost / 1e9, Qgrid=Qgrid)


def run_dp_hourly(P_fpv, P_dem, Q_in_h, V0, res, plant, cfg, Q_tur_max=None, **kw):
    """DP journalière puis réalisation horaire (rampe, Q_min, V_min, P_dispo) avec le même simulateur que le RBC."""
    n = len(P_fpv); n_days = n // 24
    Q_in_d = np.asarray(Q_in_h).reshape(n_days, 24).mean(1)
    sol = solve_dp(P_fpv, P_dem, Q_in_d, V0, res, plant, cfg, Q_tur_max, **kw)
    deficit = np.maximum(np.asarray(P_dem) - np.asarray(P_fpv), 0.0).reshape(n_days, 24)
    plan = np.zeros(n)
    for d in range(n_days):
        z = float(res.level(sol["V_d"][d]))
        plan[d * 24:(d + 1) * 24] = hourly_plan(deficit[d], z, sol["Q_d"][d] * 24.0, plant, Q_tur_max)
    out = run_rbc(P_fpv, P_dem, Q_in_h, V0, res, plant, cfg, Q_tur_max=Q_tur_max, q_plan=plan)
    out["dp"] = sol
    return out
