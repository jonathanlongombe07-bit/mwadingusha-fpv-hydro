"""Dispatch par règles (RBC), pas HORAIRE, hydro non linéaire H(z), conservation stricte du volume.

Port de bloc12_rbc.m : déficit -> débit, rampe, pré-limitation V_min, cohérence marche/arrêt,
plafond P_disponible, déversement au-delà de V_max.
"""
import numpy as np


def run_rbc(P_fpv, P_dem, Q_in, V0, res, plant, cfg, Q_tur_max=None, Q_out_extra=None, dt=3600.0, q_plan=None):
    """q_plan : débits cibles horaires (suiveur de plan, ex. issu de la DP) ; None = règle RBC."""
    H = len(P_fpv)
    Qmax = plant.Q_max if Q_tur_max is None else Q_tur_max
    Qmin, Pcap = plant.Q_min, plant.P_cap
    ramp = cfg.hydro.rampe_frac_par_pas * Qmax
    Vmin, Vmax = res.V_min, res.V_max
    v_haut = cfg.reservoir.seuil_haut_Mm3 * 1e6
    Qx = np.zeros(H) if Q_out_extra is None else Q_out_extra

    Q = np.zeros(H); u = np.zeros(H); P_h = np.zeros(H)
    P_def = np.zeros(H); P_curt = np.zeros(H); spill = np.zeros(H)
    V = np.empty(H + 1); V[0] = V0; z_hist = np.empty(H)

    for t in range(H):
        z = float(res.level(V[t])); z_hist[t] = z
        Qlim = min(Qmax, float(plant.q_for_power(Pcap, z)))
        deficit = P_dem[t] - P_fpv[t]
        if q_plan is not None:
            q = min(float(q_plan[t]), Qlim)
        elif deficit <= 0:
            q = Qmin if V[t] > v_haut else 0.0
        else:
            qn = float(plant.q_for_power(deficit, z))
            q = Qlim if qn > Qlim else (Qmin if qn < Qmin else qn)
        if t > 0:
            q = min(max(q, max(0.0, Q[t - 1] - ramp)), min(Qlim, Q[t - 1] + ramp))
        q_vmin = (V[t] - Vmin) / dt + Q_in[t] - Qx[t]            # pré-limitation V >= V_min
        if q > q_vmin:
            q = max(q_vmin, 0.0)
        if q < Qmin:
            q, u[t] = 0.0, 0.0
        else:
            u[t] = 1.0
        ph = min(float(plant.power(q, z)), Pcap)
        Q[t], P_h[t] = q, ph
        bal = P_fpv[t] + ph - P_dem[t]
        if bal < 0: P_def[t] = -bal
        else: P_curt[t] = bal
        Vn = V[t] + dt * (Q_in[t] - q - Qx[t])
        if Vn > Vmax:
            spill[t] = (Vn - Vmax) / dt; Vn = Vmax
        V[t + 1] = Vn
    return dict(Q_tur=Q, u=u, P_H=P_h, P_def=P_def, P_curt=P_curt, Q_spill=spill, V=V, z=z_hist,
                P_fpv=np.asarray(P_fpv), P_dem=np.asarray(P_dem), Q_in=np.asarray(Q_in), Q_out_extra=Qx, dt=dt)
