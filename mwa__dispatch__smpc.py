"""SMPC (MPC stochastique à scénarios) à horizon glissant H = 14 jours, N = 20 scénarios d'apport, pas de décision = 1 jour.

Formulation (MES formulations ; le détail du SMPC MATLAB (bloc13) n'est pas dans le cahier des charges fourni)
- À chaque jour d : on connaît V(d) observé, le déficit horaire (PV et charge supposés CONNUS : HYPOTHÈSE, seul l'apport est incertain).
- Décision non anticipative : le débit moyen du jour d, Q0, est COMMUN à tous les scénarios ; les jours suivants ont un recours Q_{k,s}.
- Coût journalier C_k(Q) = déficit minimal de la journée avec budget d'eau 24*Q (même fonction que la DP, hydro non linéaire à la charge z(V(d))),
  rendue convexe (enveloppe convexe inférieure) puis écrite en épigraphe : D_{k,s} >= a_j Q_{k,s} + b_j.
- Continuité : V_{k+1,s} = V_{k,s} + dt (Qin_{k,s} - Q_{k,s} - Sp_{k,s}), V >= V_min (relâché par une variable de pénalité), V <= V_max.
- Objectif : D_0 + (1/N) sum_s sum_{k>=1} D_{k,s} + M_pen * pénalité V_min - lambda_fin * E_s[V_H].
  lambda_fin (valeur terminale de l'eau, énergie évitée par Mm3) : HYPOTHÈSE, = gamma_fin x pente des courbes de coût à Q = apport moyen prévu ;
  gamma_fin < 1 (défaut 0,9) = préférence pour l'eau turbinée certaine plutôt que stockée incertaine ; gamma_fin = 1 rend le LP indifférent
  (stockage = turbinage) et le contrôleur thésaurise (constaté : déficit 252 GWh).
- Prévision d'apport (HYPOTHÈSE) : persistance du dernier apport journalier observé + incréments bootstrap de trajectoires historiques 2025
  (|t' - d| > H pour ne pas réutiliser le futur immédiat) ; calibrage sur la même année => biais optimiste (pas d'année indépendante disponible).
- Application : seul Q0 est appliqué, réalisé HORAIREMENT (affectation gloutonne + rampe/Q_min/V_min/P_dispo) avec l'apport réel.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .rbc import run_rbc
from .dp import _cost_curve, hourly_plan


def _lower_hull_lines(x, y):
    """Lignes (a, b) de l'enveloppe convexe inférieure de y(x) (y décroissant/convexe attendu)."""
    h = []
    for i in range(len(x)):
        while len(h) >= 2 and (y[h[-1]] - y[h[-2]]) * (x[i] - x[h[-1]]) >= (y[i] - y[h[-1]]) * (x[h[-1]] - x[h[-2]]):
            h.pop()
        h.append(i)
    return [((y[j] - y[i]) / (x[j] - x[i]), y[i] - (y[j] - y[i]) / (x[j] - x[i]) * x[i]) for i, j in zip(h[:-1], h[1:])]


def make_scenarios(Q_in_d, d, H, N, rng):
    """Matrice (N, H) de prévisions d'apport pour les jours d..d+H-1 (persistance + incréments bootstrap)."""
    n = len(Q_in_d)
    base = Q_in_d[d - 1] if d > 0 else Q_in_d[0]
    cand = [t for t in range(0, n - H - 1) if abs(t - d) > H]
    if len(cand) < N: cand = list(range(0, n - H - 1))
    ts = rng.choice(cand, size=N, replace=len(cand) < N)
    return np.array([base + (Q_in_d[t + 1:t + 1 + H] - Q_in_d[t]) for t in ts])


def solve_step(deficit_days, z, V, Qin_scen, res, plant, Qmax, H, N, Qgrid, M_pen=1e4, gamma_fin=0.9):
    """Résout le LP SMPC ; retourne Q0 (m3/s) et diagnostics. Unités : Wh pour les déficits, Mm3 pour les volumes."""
    K = deficit_days.shape[0]
    lines = []
    cost_avg = np.zeros(len(Qgrid))
    for k in range(K):
        c = _cost_curve(deficit_days[k], z, Qgrid, plant, Qmax)
        cost_avg += c / K
        lines.append(_lower_hull_lines(Qgrid, c / 1e9))       # GWh
    Qbar = float(np.clip(Qin_scen.mean(), plant.Q_min, Qmax))
    slope = np.gradient(cost_avg / 1e9, Qgrid)
    lam = max(-float(np.interp(Qbar, Qgrid, slope)) * 1e6 / 86400.0, 0.0)   # GWh évités par Mm3 (1 m3/s pendant 1 j = 0,0864 Mm3)
    # variables : Q0 | pour s: Q_{k,s} (k=1..K-1) | D0 | D_{k,s} (k=1..K-1) | V_{k,s} (k=1..K) | Sp_{k,s} (k=0..K-1) | slack_{k,s} (k=1..K)
    nq = K - 1
    iQ0 = 0
    iQ = lambda s, k: 1 + s * nq + (k - 1)
    iD0 = 1 + N * nq
    iD = lambda s, k: iD0 + 1 + s * nq + (k - 1)
    base = iD0 + 1 + N * nq
    iV = lambda s, k: base + s * K + (k - 1)
    base2 = base + N * K
    iSp = lambda s, k: base2 + s * K + k
    base3 = base2 + N * K
    iSl = lambda s, k: base3 + s * K + (k - 1)
    n = base3 + N * K
    cvec = np.zeros(n); cvec[iD0] = 1.0
    lb = np.zeros(n); ub = np.full(n, np.inf)
    ub[iQ0] = Qmax
    k_dt = 86400.0 / 1e6
    rows, cols, vals, rhs = [], [], [], []; r = 0
    eq_r, eq_c, eq_v, eq_b = [], [], [], []; er = 0
    for a, b in lines[0]:                                    # D0 >= a Q0 + b
        rows += [r, r]; cols += [iD0, iQ0]; vals += [-1.0, a]; rhs.append(-b); r += 1
    for s in range(N):
        for k in range(1, K):
            ub[iQ(s, k)] = Qmax
            cvec[iD(s, k)] = 1.0 / N
            for a, b in lines[k]:
                rows += [r, r]; cols += [iD(s, k), iQ(s, k)]; vals += [-1.0, a]; rhs.append(-b); r += 1
        for k in range(K):                                   # continuité : V_{k+1} - V_k + dt Q + dt Sp = dt Qin
            Vprev = None if k == 0 else iV(s, k)
            Qvar = iQ0 if k == 0 else iQ(s, k)
            eq_r += [er, er, er]; eq_c += [iV(s, k + 1) if k + 1 <= K else None, Qvar, iSp(s, k)] ; eq_v += [1.0, k_dt, k_dt]
            if Vprev is not None:
                eq_r.append(er); eq_c.append(Vprev); eq_v.append(-1.0)
                eq_b.append(k_dt * Qin_scen[s, k])
            else:
                eq_b.append(k_dt * Qin_scen[s, k] + V / 1e6)
            er += 1
        for k in range(1, K + 1):
            lb[iV(s, k)] = -1e3; ub[iV(s, k)] = res.V_max / 1e6
            rows += [r, r]; cols += [iV(s, k), iSl(s, k)]; vals += [-1.0, -1.0]; rhs.append(-res.V_min / 1e6); r += 1   # V + slack >= V_min
            cvec[iSl(s, k)] = M_pen / N
        cvec[iV(s, K)] -= gamma_fin * lam / N                           # valeur terminale
    A_ub = sp.csr_matrix((vals, (rows, cols)), shape=(r, n))
    A_eq = sp.csr_matrix((eq_v, (eq_r, eq_c)), shape=(er, n))
    sol = linprog(cvec, A_ub=A_ub, b_ub=rhs, A_eq=A_eq, b_eq=eq_b, bounds=list(zip(lb, ub)), method="highs")
    if sol.status != 0:
        return None, dict(status=sol.status, lam=lam)
    return float(sol.x[iQ0]), dict(status=0, lam=lam, obj=float(sol.fun))


def run_smpc(P_fpv, P_dem, Q_in_h, V0, res, plant, cfg, Q_tur_max=None, H_days=14, N=20, seed=0, nQ=24, gamma_fin=0.9, verbose=False):
    n = len(P_fpv); n_days = n // 24
    Qmax = plant.Q_max if Q_tur_max is None else Q_tur_max
    Q_in_d = np.asarray(Q_in_h).reshape(n_days, 24).mean(1)
    deficit = np.maximum(np.asarray(P_dem) - np.asarray(P_fpv), 0.0).reshape(n_days, 24)
    Qgrid = np.concatenate([[0.0], np.linspace(plant.Q_min, Qmax, nQ - 1)])
    rng = np.random.default_rng(seed)
    V = V0; plan = np.zeros(n); Vlist = []; nfail = 0; lams = []
    out_parts = []
    for d in range(n_days):
        K = min(H_days, n_days - d)
        z = float(res.level(V))
        if K >= 2:
            scen = make_scenarios(Q_in_d, d, K, N, rng)
            q0, info = solve_step(deficit[d:d + K], z, V, scen, res, plant, Qmax, K, N, Qgrid, gamma_fin=gamma_fin)
            lams.append(info.get("lam", np.nan))
        else:
            q0, info = None, dict(status=-1)
        if q0 is None:
            nfail += 1
            q0 = float(np.clip(Q_in_d[max(d - 1, 0)], 0.0, Qmax))     # repli : persistance de l'apport
        plan_d = hourly_plan(deficit[d], z, q0 * 24.0, plant, Qmax)
        sl = slice(d * 24, (d + 1) * 24)
        part = run_rbc(P_fpv[sl], P_dem[sl], Q_in_h[sl], V, res, plant, cfg, Q_tur_max=Qmax, q_plan=plan_d)
        V = float(part["V"][-1]); out_parts.append(part); plan[sl] = part["Q_tur"]
    # assemblage de la série complète
    keys = ["Q_tur", "u", "P_H", "P_def", "P_curt", "Q_spill", "z", "P_fpv", "P_dem", "Q_in", "Q_out_extra"]
    out = {k: np.concatenate([p[k] for p in out_parts]) for k in keys}
    out["V"] = np.concatenate([p["V"][:-1] for p in out_parts] + [[out_parts[-1]["V"][-1]]])
    out["dt"] = 3600.0
    out["smpc"] = dict(n_echecs_LP=nfail, lambda_moyen_GWh_par_Mm3=float(np.nanmean(lams)) if lams else np.nan, H_days=H_days, N=N, seed=seed)
    return out
