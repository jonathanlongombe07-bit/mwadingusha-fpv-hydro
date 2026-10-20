"""Génère toutes les figures (PNG 300 dpi + PDF) et les tableaux mensuels de l'annexe C.

Usage : python run_figures.py [--out figures] [--charge ferme|profil] [--kpi resultats/kpi_dp_vs_rbc.csv]
Recalcule les simulations 60 MW (hydro seul / hybride, RBC et DP) ; la figure surcharge lit le CSV de run_dp.py.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from mwa.config import load_config
from mwa import forcings, data, loadprofile, hydroclimat, figures as fg
from mwa.dispatch.rbc import run_rbc
from mwa.dispatch.dp import run_dp_hourly


def main(out="figures", mode_charge="ferme", kpi_csv="resultats/kpi_dp_vs_rbc.csv", P_MW=60.0):
    cfg = load_config(); F = forcings.build(cfg); n = len(F.idx)
    d = Path(out); d.mkdir(exist_ok=True, parents=True); fg.style_publication()
    nasa = data.load_nasa()
    prof = None if mode_charge == "ferme" else F.profile
    Pd = loadprofile.pdem_series(F.idx, P_MW * 1e6, prof); V0 = float(F.hydro_d["V_obs"].iloc[0])
    runs = {}
    for cas, pf, qi in (("hydro_seul", np.zeros(n), F.Q_in_h), ("hybride", F.fpv["P_fpv"].values, F.Q_in_h + F.Q_evap_saved_h)):
        runs[(cas, "RBC")] = run_rbc(pf, Pd, qi, V0, F.res, F.plant, cfg)
        runs[(cas, "DP")] = run_dp_hourly(pf, Pd, qi, V0, F.res, F.plant, cfg)
    for k, o in runs.items():
        o["P_fpv"] = np.asarray(o["P_fpv"])
    made = []
    made += [fg.fig_forcages_meteo(nasa, F, d), fg.fig_fpv(F, d), fg.fig_evaporation(F, d), fg.fig_cycle_annuel(F, d),
             fg.fig_reservoir_hist(F, d), fg.fig_annexeC_debits(F, d), fg.fig_spearman(nasa, F, d)]
    fg.tables_mensuelles(nasa, F, d)
    made += [fg.fig_comparaison_globale(runs, d, f"{P_MW:.0f} MW"),
             fg.fig_reservoir_dyn(runs, "hydro_seul", F.idx, F.res, d, "fig_p3_02_reservoir_A", "Scénario A (hydro seul)"),
             fg.fig_reservoir_dyn(runs, "hybride", F.idx, F.res, d, "fig_p3_03_reservoir_B", "Scénario B (hybride)"),
             fg.fig_deficit_series(runs, F.idx, d), fg.fig_boxplot_et_duree(runs, F.idx, d),
             fg.fig_chap6_hydro(runs, F.idx, d), fg.fig_chap6_dispatch(runs, F.idx, d)]
    r = Path(kpi_csv).parent
    if (r / "matrice_kpi_4strategies.csv").exists():                      # sorties de run_all.py / run_sensibilite.py
        M = pd.read_csv(r / "matrice_kpi_4strategies.csv"); S = pd.read_csv(r / "kpi_par_scenario.csv")
        sc, masks, info = hydroclimat.build(F)
        made += [fg.fig_matrice_strategies(M, d), fg.fig_strat_60_75(M, d), fg.fig_surcharge4(M, d), fg.fig_scen_effectifs(sc, info, d),
                 fg.fig_scen_deficit(S, d, 75.0), fg.fig_scen_stress(S, d),
                 fg.fig_scen_classement(S, d), fg.fig_scen_robustesse(S, d), fg.fig_daily_strategies(pd.read_csv(r / "journalier_base.csv"), d, 75.0)]
        if (r / "sensibilite_couverture.csv").exists(): made.append(fg.fig_couverture(pd.read_csv(r / "sensibilite_couverture.csv"), d))
        if (r / "potentiel_fpv.csv").exists(): made.append(fg.fig_potentiel(pd.read_csv(r / "potentiel_fpv.csv"), d))
        if (r / "sobol_indices.csv").exists(): made.append(fg.fig_sobol(pd.read_csv(r / "sobol_indices.csv"), d))
    elif Path(kpi_csv).exists():
        made.append(fg.fig_surcharge(pd.read_csv(kpi_csv), d))
    print(f"{len(made)} figures écrites dans {d}/ (PNG + PDF) + 3 tableaux mensuels annexe C")
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="figures")
    ap.add_argument("--charge", default="ferme", choices=["ferme", "profil"]); ap.add_argument("--kpi", default="resultats/kpi_dp_vs_rbc.csv")
    a = ap.parse_args(); main(a.out, a.charge, a.kpi)
