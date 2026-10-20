"""Indicateurs (port de bloc21 avec corrections) : eau économisée par DIFFÉRENCE de deux simulations."""
import numpy as np


def kpi(out, res, cfg, out_ref=None):
    dt = out["dt"]; T = len(out["P_dem"]); GWh = dt / 3.6e12
    V = out["V"]; Vc = cfg.reservoir.V_critique_Mm3 * 1e6
    E_dem = out["P_dem"].sum() * GWh; E_def = out["P_def"].sum() * GWh
    k = {
        "E_fpv_GWh": out["P_fpv"].sum() * GWh, "E_H_GWh": out["P_H"].sum() * GWh,
        "E_dem_GWh": E_dem, "E_def_GWh": E_def, "E_curt_GWh": out["P_curt"].sum() * GWh,
        "R": 1 - E_def / max(E_dem, 1e-9),
        "heures_deficit_pct": 100 * float(np.mean(out["P_def"] > 1e3)),
        "V_min_Mm3": float(V.min() / 1e6), "V_fin_Mm3": float(V[-1] / 1e6),
        "jours_V_critique": float(np.sum(V[:-1] < Vc) / 24.0),
        "eau_turbinee_Mm3": out["Q_tur"].sum() * dt / 1e6, "deverse_Mm3": out["Q_spill"].sum() * dt / 1e6,
    }
    if out_ref is not None:
        k["eau_economisee_Mm3"] = (out_ref["Q_tur"].sum() - out["Q_tur"].sum()) * dt / 1e6
        k["deficit_evite_GWh"] = out_ref["P_def"].sum() * GWh - E_def
        k["V_fin_gain_Mm3"] = (V[-1] - out_ref["V"][-1]) / 1e6
    return k


def kpi_ext(out, res, cfg, Q_in_tot_Mm3=None):
    """Indicateurs du cahier des charges (§9) : E_PV, E_H, E_def, V_eco_eau, V_min_obs, N_crit, R, R_comp.

    K_H effectif = sum(P_H)/sum(Q_tur) (W par m3/s) : l'hydro étant non linéaire, K_H n'est pas une constante ;
    V_eco_eau = sum(P_fpv)/K_H * dt (formule du cahier) est donc donnée avec ce K_H effectif ; la méthode par
    DIFFÉRENCE de deux simulations (kpi(..., out_ref)) reste la référence pour l'eau réellement économisée.
    N_crit est donné avec le seuil V_critique_Mm3 (183,5 Mm3, notre choix) ET avec 0,30 x V_max (formule du cahier, 330 Mm3).
    """
    dt = out["dt"]; T = len(out["P_dem"]); GWh = dt / 3.6e12
    k = kpi(out, res, cfg)
    KH = out["P_H"].sum() / max(out["Q_tur"].sum(), 1e-9)
    v_eco = out["P_fpv"].sum() / KH * dt / 1e6
    q_tot = (out["Q_in"].sum() * dt / 1e6) if Q_in_tot_Mm3 is None else Q_in_tot_Mm3
    V = out["V"][:-1]
    n_crit_abs = float(np.sum(V < cfg.reservoir.V_critique_Mm3 * 1e6)); n_crit_30 = float(np.sum(V < 0.30 * res.V_max))
    f_def = 1 - k["E_def_GWh"] / max(k["E_dem_GWh"], 1e-9)
    f_eau = min(1.0, v_eco / max(q_tot, 1e-9)); f_crit = 1 - n_crit_abs / T
    k.update(K_H_eff_MW_par_m3s=KH / 1e6, V_eco_eau_Mm3=v_eco, V_min_obs_Mm3=float(V.min() / 1e6),
             N_crit_h=n_crit_abs, N_crit_30Vmax_h=n_crit_30, R_comp=0.5 * f_def + 0.2 * f_eau + 0.3 * f_crit)
    return k


def daily_frame(out, idx, T_cell=None):
    """Résultats journaliers (moyennes MW / m3/s ; V de fin de jour)."""
    import pandas as pd
    n = len(idx); nd = n // 24; d = idx[::24][:nd].normalize()
    m = lambda x: np.asarray(x)[:nd * 24].reshape(nd, 24).mean(1)
    df = pd.DataFrame({"P_FPV_MW": m(out["P_fpv"]) / 1e6, "P_H_MW": m(out["P_H"]) / 1e6, "P_dem_MW": m(out["P_dem"]) / 1e6,
                       "P_def_MW": m(out["P_def"]) / 1e6, "P_curt_MW": m(out["P_curt"]) / 1e6, "Q_tur_m3s": m(out["Q_tur"]),
                       "Q_spill_m3s": m(out["Q_spill"]), "Q_in_m3s": m(out["Q_in"]), "V_fin_Mm3": np.asarray(out["V"])[24::24][:nd] / 1e6},
                      index=d)
    if T_cell is not None:
        df["T_cell_C"] = m(T_cell)
    df.index.name = "date"
    return df
