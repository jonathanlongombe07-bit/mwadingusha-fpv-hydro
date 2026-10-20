"""MILP horaire à prévision parfaite, hydro LINÉAIRE PAR MORCEAUX, rejouée ensuite en NON LINÉAIRE.

Port de bloc11_milp.m avec MES formulations (le MATLAB utilisait P = K_H * Q, linéaire, pas journalier).

Variables par heure t (+ L_t >= 0 : relâchement de V_min, pénalisé M_vmin, utile en stress) : Q_t (m3/s), u_t binaire (groupe(s) en marche), P_t (MW hydro livrés),
Pd_t (MW déficit), Pc_t (MW écrêté), S_t (m3/s déversé), V_t (Mm3).
Contraintes
  Q_min u <= Q <= Q_max u                      (marche/arrêt)
  P_t <= a_k Q_t + b_k u_t   pour chaque arête k de l'ENVELOPPE CONCAVE de P(Q ; z_t)   (hydro linéarisée par morceaux)
  P_t <= P_dispo
  P_t + P_fpv + Pd_t - Pc_t = P_dem           (équilibre énergétique)
  V_{t+1} = V_t + dt (Q_in - Q - S - Q_extra)  (continuité)
  V_min <= V <= V_max ;  |Q_{t+1}-Q_t| <= rampe
Objectif : M_def*Pd + M_curt*Pc + M_spill*S   (+ contrainte terminale optionnelle V_T >= V_ref)

Linéarisation de la charge : le niveau z_t est FIXÉ à une trajectoire de référence (itération : RBC -> MILP -> rejeu
non linéaire -> MILP ...). L'enveloppe concave SURESTIME la puissance là où P(Q) n'est pas concave (marches du nombre de
groupes) : le rejeu non linéaire mesure l'écart ("coût de la linéarisation").
"""
import time
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

from .rbc import run_rbc


def _upper_hull(x, y):
    """Enveloppe concave supérieure (liste d'indices) de points triés en x."""
    h = []
    for i in range(len(x)):
        while len(h) >= 2 and (y[h[-1]] - y[h[-2]]) * (x[i] - x[h[-1]]) <= (y[i] - y[h[-1]]) * (x[h[-1]] - x[h[-2]]):
            h.pop()
        h.append(i)
    return h


def pwl_lines(plant, z, n_bp=14, Qmax=None):
    """Arêtes (a, b) en MW et MW/(m3/s) de l'enveloppe concave de P(Q) sur [Q_min, Q_max], à la charge z.
    Écriture P <= a Q + b u (u=1 en marche)."""
    q = np.linspace(plant.Q_min, plant.Q_max if Qmax is None else Qmax, n_bp)
    p = np.minimum(plant.power(q, z), plant.P_cap) / 1e6
    h = _upper_hull(q, p)
    lines = []
    for i, j in zip(h[:-1], h[1:]):
        a = (p[j] - p[i]) / (q[j] - q[i]); b = p[i] - a * q[i]
        lines.append((a, b))
    return lines


def solve_milp_window(P_fpv, P_dem, Q_in, Qx, V0, z_ref, res, plant, cfg, Q_max_eff, V_end_min=None,
                      M_def=1e4, M_curt=1.0, M_spill=1e-2, M_vmin=1e7, time_limit=120.0, gap=1e-4, n_bp=14):
    T = len(P_fpv); dt = 3600.0
    pf = np.asarray(P_fpv) / 1e6; pdm = np.asarray(P_dem) / 1e6
    # indices
    iQ = np.arange(T); iu = T + iQ; iP = 2 * T + iQ; iD = 3 * T + iQ; iC = 4 * T + iQ; iS = 5 * T + iQ; iV = 6 * T + np.arange(T + 1); iL = 7 * T + 1 + np.arange(T + 1)
    n = 8 * T + 2
    c = np.zeros(n); c[iD] = M_def; c[iC] = M_curt; c[iS] = M_spill; c[iL] = M_vmin
    lb = np.zeros(n); ub = np.full(n, np.inf)
    ub[iQ] = Q_max_eff; ub[iu] = 1; ub[iP] = plant.P_cap / 1e6
    lb[iV] = -1e3; ub[iV] = res.V_max / 1e6
    lb[iV[0]] = ub[iV[0]] = V0 / 1e6
    integrality = np.zeros(n); integrality[iu] = 1
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0

    def add(entries, l, h):
        nonlocal r
        for cc, vv in entries: rows.append(r); cols.append(cc); vals.append(vv)
        lo.append(l); hi.append(h); r += 1

    for t in range(T):
        add([(iQ[t], 1.0), (iu[t], -Q_max_eff)], -np.inf, 0.0)               # Q <= Qmax u
        add([(iQ[t], 1.0), (iu[t], -plant.Q_min)], 0.0, np.inf)              # Q >= Qmin u
        for a, b in pwl_lines(plant, float(z_ref[t]), n_bp, Q_max_eff):
            add([(iP[t], 1.0), (iQ[t], -a), (iu[t], -b)], -np.inf, 0.0)      # P <= a Q + b u
        add([(iP[t], 1.0), (iD[t], 1.0), (iC[t], -1.0)], pdm[t] - pf[t], pdm[t] - pf[t])   # équilibre
        k = dt / 1e6                                                          # continuité (Mm3)
        e = k * (Q_in[t] - Qx[t])
        add([(iV[t + 1], 1.0), (iV[t], -1.0), (iQ[t], k), (iS[t], k)], e, e)
    for t in range(T + 1):                                                    # V + L >= V_min (relâché si infaisable : stress)
        add([(iV[t], 1.0), (iL[t], 1.0)], res.V_min / 1e6, np.inf)
    ramp = cfg.hydro.rampe_frac_par_pas * Q_max_eff
    for t in range(T - 1):
        add([(iQ[t + 1], 1.0), (iQ[t], -1.0)], -ramp, ramp)
    if V_end_min is not None:
        lb[iV[-1]] = max(lb[iV[-1]], V_end_min / 1e6)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, n))
    t0 = time.time()
    sol = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=integrality, bounds=Bounds(lb, ub),
               options=dict(time_limit=time_limit, mip_rel_gap=gap, presolve=True))
    info = dict(status=int(sol.status), success=bool(sol.success), obj=float(sol.fun) if sol.x is not None else np.nan,
                gap=float(getattr(sol, "mip_gap", np.nan)), time_s=time.time() - t0, message=sol.message)
    if sol.x is None:
        return None, info
    x = sol.x
    return dict(Q=x[iQ], u=np.round(x[iu]), P=x[iP] * 1e6, Pd=x[iD] * 1e6, V=x[iV] * 1e6), info


def run_milp(P_fpv, P_dem, Q_in_h, V0, res, plant, cfg, Q_tur_max=None, Q_out_extra=None, n_iter=2,
             window_days=None, V_ref_traj=None, V_end_ref=None, time_limit=300.0, gap=1e-3, verbose=False):
    """MILP + rejeu non linéaire. window_days=None : un seul bloc (single-shot, comme le MATLAB ; trop lent en horaire sur 1 an).
    window_days=k : fenêtres de k jours à prévision parfaite DANS la fenêtre ; V_end_ref (H+1) impose en fin de fenêtre
    V >= V_end_ref[e] - tol (ex. trajectoire de la DP journalière : évite que la fenêtre vide le lac).
    V_ref_traj : trajectoire initiale de volume (H+1) pour fixer la charge ; sinon issue du RBC."""
    n = len(P_fpv); Qmax = plant.Q_max if Q_tur_max is None else Q_tur_max
    Qx = np.zeros(n) if Q_out_extra is None else np.asarray(Q_out_extra, float)
    Qmax_eff = min(Qmax, plant.Q_max)
    if V_ref_traj is None:
        V_ref_traj = run_rbc(P_fpv, P_dem, Q_in_h, V0, res, plant, cfg, Q_tur_max=Qmax, Q_out_extra=Qx)["V"]
    infos = []; out = None
    W = n if window_days is None else window_days * 24
    for it in range(n_iter):
        z_ref = res.level(V_ref_traj[:-1])
        plan = np.zeros(n)
        Vcur = V0
        for s in range(0, n, W):
            e = min(s + W, n)
            Vend = None if V_end_ref is None else float(V_end_ref[e]) - 0.5e6
            sol, info = solve_milp_window(P_fpv[s:e], P_dem[s:e], Q_in_h[s:e], Qx[s:e], Vcur, z_ref[s:e], res, plant, cfg,
                                          Qmax_eff, V_end_min=Vend, time_limit=time_limit, gap=gap)
            if sol is None and Vend is not None:     # cible terminale inatteignable depuis Vcur : on la relâche
                info["relache_V_end"] = True
                sol, info2 = solve_milp_window(P_fpv[s:e], P_dem[s:e], Q_in_h[s:e], Qx[s:e], Vcur, z_ref[s:e], res, plant, cfg,
                                               Qmax_eff, V_end_min=None, time_limit=time_limit, gap=gap)
                info.update({k: v for k, v in info2.items() if k != "message"}); info["message"] = info2["message"]
            infos.append(info)
            if sol is None:
                raise RuntimeError(f"MILP sans solution (fenêtre {s}-{e}) : {info['message']}")
            plan[s:e] = sol["Q"]
            # volume de départ de la fenêtre suivante : celui du rejeu non linéaire de ce bloc
            if e < n:
                tmp = run_rbc(P_fpv[s:e], P_dem[s:e], Q_in_h[s:e], Vcur, res, plant, cfg, Q_tur_max=Qmax,
                              Q_out_extra=Qx[s:e], q_plan=sol["Q"])
                Vcur = float(tmp["V"][-1])
        out = run_rbc(P_fpv, P_dem, Q_in_h, V0, res, plant, cfg, Q_tur_max=Qmax, Q_out_extra=Qx, q_plan=plan)
        V_ref_traj = out["V"]
        if verbose:
            print(f"  MILP it.{it + 1}: déficit rejoué = {out['P_def'].sum() * 3600 / 3.6e12:.2f} GWh ; "
                  f"temps {sum(i['time_s'] for i in infos[-max(1, n // W):]):.1f} s")
    out["milp"] = dict(infos=infos, n_iter=n_iter, window_days=window_days)
    return out
