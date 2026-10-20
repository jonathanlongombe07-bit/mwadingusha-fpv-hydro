"""Tests formels de cohérence (port de bloc18_validation.m, pas horaire, V de longueur H+1)."""
import numpy as np


def validate(out, res, plant, tol_mass=1.0, tol_energy=1e3):
    dt = out["dt"]; V = out["V"]
    exp = V[:-1] + dt * (out["Q_in"] - out["Q_tur"] - out["Q_spill"] - out["Q_out_extra"])
    mass = float(np.max(np.abs(V[1:] - exp)))
    energy = float(np.max(np.abs(out["P_fpv"] + out["P_H"] + out["P_def"] - out["P_curt"] - out["P_dem"])))
    below = float(np.min(V) - res.V_min)
    checks = {
        "V<=V_max": bool(np.all(V <= res.V_max + 1.0)),
        "Q>=0": bool(np.all(out["Q_tur"] >= -1e-9)),
        "Q<=Q_max": bool(np.all(out["Q_tur"] <= plant.Q_max + 1e-9)),
        "P_H<=P_dispo": bool(np.all(out["P_H"] <= plant.P_cap + 1e-6)),
        "P_def>=0": bool(np.all(out["P_def"] >= 0)), "P_curt>=0": bool(np.all(out["P_curt"] >= 0)),
        "u=0=>Q=0": bool(np.all(out["Q_tur"][out["u"] == 0] < 1e-9)),
        "u=1=>Q>=Qmin": bool(np.all(out["Q_tur"][out["u"] == 1] >= plant.Q_min - 1e-9)),
    }
    q_in_tot = out["Q_in"].sum() * dt / 1e6
    resid = q_in_tot - (out["Q_tur"].sum() + out["Q_spill"].sum() + out["Q_out_extra"].sum()) * dt / 1e6 - (V[-1] - V[0]) / 1e6
    return dict(mass_residual_m3=mass, energy_residual_W=energy, V_below_Vmin_m3=max(-below, 0.0),
                checks=checks, global_residual_Mm3=float(resid),
                ok=bool(mass < tol_mass and energy < tol_energy and all(checks.values())))


def formal_tests(out, res, plant, cfg, tol_global=0.01):
    """6 tests formels du cahier des charges (§10). Les définitions de (e) et (f) ne sont pas détaillées dans le cahier : HYPOTHÈSE ci-dessous.

    (a) conservation de masse pas à pas : max|V(t+1) - [V(t) + dt (Qin - Qtur - Qspill - Qextra)]| < 1 m3
    (b) conservation d'énergie : max|P_fpv + P_H + P_def - P_curt - P_dem| < 1e3 W
    (c) contraintes physiques : V <= V_max ; 0 <= Q <= Q_max ; 0 <= P_H <= P_dispo ; P_def, P_curt >= 0.
        Le respect de V >= V_min est rapporté à part (c_Vmin) : il peut être impossible si l'apport NET (évaporation incluse) est négatif.
    (d) marche/arrêt : u=0 => Q=0 ; u=1 => Q >= Q_min
    (e) cohérence temporelle (HYPOTHÈSE) : V recalculé par intégration depuis V0 avec les débits de sortie = V rapporté (dérive finale < 1 m3 x nb pas)
        ET |V(t+1)-V(t)| <= dt (max|Qin| + Q_max) à chaque pas
    (f) bilan global (HYPOTHÈSE) : |apport total - turbiné - déversé - extra - dV| / apport total < 1 %
    """
    dt = out["dt"]; V = np.asarray(out["V"]); T = len(out["P_dem"])
    step = dt * (out["Q_in"] - out["Q_tur"] - out["Q_spill"] - out["Q_out_extra"])
    a = float(np.max(np.abs(V[1:] - (V[:-1] + step))))
    b = float(np.max(np.abs(out["P_fpv"] + out["P_H"] + out["P_def"] - out["P_curt"] - out["P_dem"])))
    c_ok = bool(np.all(V <= res.V_max + 1.0) and np.all(out["Q_tur"] >= -1e-9) and np.all(out["Q_tur"] <= plant.Q_max + 1e-9)
                and np.all(out["P_H"] >= -1e-6) and np.all(out["P_H"] <= plant.P_cap + 1e-6)
                and np.all(out["P_def"] >= 0) and np.all(out["P_curt"] >= 0))
    below = np.maximum(res.V_min - V[1:], 0.0)
    forced = bool(np.all(out["Q_tur"][below > 1.0] < 1e-9)) if np.any(below > 1.0) else True
    d_ok = bool(np.all(out["Q_tur"][out["u"] == 0] < 1e-9) and np.all(out["Q_tur"][out["u"] == 1] >= plant.Q_min - 1e-9))
    Vint = V[0] + np.concatenate([[0.0], np.cumsum(step)])
    drift = float(np.max(np.abs(Vint - V)))
    bound = float(np.max(np.abs(np.diff(V)))), float(dt * (np.max(np.abs(out["Q_in"])) + plant.Q_max))
    e_ok = bool(drift < 1.0 * T and bound[0] <= bound[1])
    q_in_tot = out["Q_in"].sum() * dt
    f_rel = float(abs(q_in_tot - (out["Q_tur"] + out["Q_spill"] + out["Q_out_extra"]).sum() * dt - (V[-1] - V[0])) / max(abs(q_in_tot), 1.0))
    return {
        "a_masse": (a < 1.0, a), "b_energie": (b < 1e3, b), "c_physique": (c_ok, float(below.max())),
        "c_Vmin": (forced, float(below.max()) / 1e6), "d_marche_arret": (d_ok, 0.0),
        "e_temporelle": (e_ok, drift), "f_bilan_global": (f_rel < tol_global, f_rel)}
