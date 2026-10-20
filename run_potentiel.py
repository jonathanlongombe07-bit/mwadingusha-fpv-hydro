"""Étude du POTENTIEL FPV sur la retenue en fonction du taux de couverture (port de bloc22_potentiel_fpv.m, avec productible horaire réel).

Pour chaque f_cov (part de la surface du lac au volume plancher V_min, S_min = 35,1 km2 avec Feuil16) :
  surface, puissance DC, nombre de modules ; énergie annuelle 2025 (8 760 h NASA, modèle pvlib) en deux variantes d'évacuation :
    (A) onduleurs proportionnels (DC/AC = 1,29 conservé)   (B) évacuation FIXE 75 MW (plafond du cahier)
  évaporation évitée (Mm3/an) et son équivalent énergie hydro ; part de la surface du lac à S_moy et S_max observés en 2025 ;
  bénéfice DISPATCH sur 11 mois (déficit RBC et DP à 60 et 75 MW) pour les deux variantes.
Usage : python run_potentiel.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

from mwa.config import load_config, replace
from mwa import forcings, loadprofile, data
from mwa.fpv import simulate_fpv
from mwa.dispatch.rbc import run_rbc
from mwa.dispatch.dp import run_dp_hourly


def main(out="resultats"):
    cfg = load_config(); F = forcings.build(cfg); n = len(F.idx)
    nasa_y = data.load_nasa()                                   # 8 760 h
    S_min = float(F.res.surface(F.res.V_min)) / 1e6             # km2
    S_d = F.res.surface(F.hydro_d["V_obs"].values) / 1e6
    S_moy, S_max = float(S_d.mean()), float(S_d.max())
    p = cfg.fpv; A_per_MWp_km2 = (1e6 / p.P_mod) * p.A_mod * p.kmarge / 1e6    # km2 par MWp (modules + marge)
    N0 = p.N_mod; V0 = float(F.hydro_d["V_obs"].iloc[0])
    KH = 1.011                                                  # MW par m3/s (K_H effectif des simulations)
    f_list = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    print(f"S_min = {S_min:.2f} km2 (V_min {cfg.reservoir.V_min_Mm3} Mm3) ; S_moy 2025 = {S_moy:.1f} ; S_max 2025 = {S_max:.1f} ; {A_per_MWp_km2 * 100:.2f} ha/MWp")
    rows = []
    for f in f_list:
        A_km2 = f * S_min; P_MWp = A_km2 / A_per_MWp_km2; N = P_MWp * 1e6 / p.P_mod
        row = dict(f_cov_pct=100 * f, A_km2=A_km2, P_DC_MWp=P_MWp, N_mod=N, f_sur_S_moy_pct=100 * A_km2 / S_moy, f_sur_S_max_pct=100 * A_km2 / S_max)
        for var, cfg_v in (("A_proportionnel", cfg), ("B_fixe75", replace(cfg, fpv={"P_AC": 75e6 * N0 / N}))):
            fy, E_open, _ = simulate_fpv(nasa_y, cfg_v, N_mod=N)
            E_gwh = fy["P_fpv"].sum() * 3600 / 3.6e12; ecr = fy["P_ecrete"].sum() * 3600 / 3.6e12
            row[f"E_an_GWh_{var}"] = E_gwh; row[f"E_ecretee_GWh_{var}"] = ecr
            row[f"kWh_par_kWp_{var}"] = E_gwh * 1e6 / (P_MWp * 1e3); row[f"FC_pct_{var}"] = 100 * E_gwh * 1e3 / (P_MWp * 8760)
            pf = fy["P_fpv"].reindex(F.idx).values
            sc = N / N0
            saved = F.Q_evap_saved_h * sc
            for P in (60.0, 75.0):
                Pd = loadprofile.pdem_series(F.idx, P * 1e6, None); qi = F.Q_in_h + saved
                for s in ("RBC", "DP"):
                    o = run_rbc(pf, Pd, qi, V0, F.res, F.plant, cfg) if s == "RBC" else run_dp_hourly(pf, Pd, qi, V0, F.res, F.plant, cfg)
                    row[f"def_GWh_{P:.0f}MW_{s}_{var}"] = o["P_def"].sum() * 3600 / 3.6e12
            if var == "A_proportionnel":
                evap_an = (cfg.evap.r_evap * min(A_km2 * 1e6, 1e12) * E_open.sum() / 1000.0) / 1e6      # Mm3/an (E_open en mm/j)
                row["evap_evitee_Mm3_an"] = evap_an; row["evap_equiv_GWh_hydro"] = evap_an * 1e6 / 3600 * KH / 1e3
        rows.append(row)
        print(f"  f={100 * f:4.0f} %  {P_MWp:7.1f} MWp  E_A {row['E_an_GWh_A_proportionnel']:7.1f}  E_B {row['E_an_GWh_B_fixe75']:6.1f} GWh/an", flush=True)
    D = pd.DataFrame(rows); Path(out).mkdir(exist_ok=True); D.to_csv(Path(out) / "potentiel_fpv.csv", index=False)
    ref = simulate_fpv(nasa_y, cfg)[0]
    print(f"\nRéférence actuelle : {N0} modules, {p.P_DC / 1e6:.2f} MWp, {N0 * p.A_mod * p.kmarge / 1e6:.3f} km2 = {100 * N0 * p.A_mod * p.kmarge / 1e6 / S_min:.2f} % de S_min ;"
          f" {ref['P_fpv'].sum() * 3600 / 3.6e12:.1f} GWh/an (8 760 h)")
    pd.set_option("display.width", 250)
    print(D.round(1).to_string(index=False))
    return D


if __name__ == "__main__":
    main()
