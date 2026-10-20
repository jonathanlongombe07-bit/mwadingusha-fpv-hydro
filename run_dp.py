"""Étape 3 : DP (borne à prévision parfaite) vs RBC, hydro seul vs hybride, toutes consignes, cas de base.

Usage : python run_dp.py [--out resultats]
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from mwa.config import load_config
from mwa import forcings, scenarios, loadprofile
from mwa.dispatch.rbc import run_rbc
from mwa.dispatch.dp import run_dp_hourly
from mwa.validation import validate
from mwa.indicators import kpi


def main(out_dir="resultats", mode_charge="ferme"):
    cfg = load_config(); F = forcings.build(cfg); n = len(F.idx)
    V0 = float(F.hydro_d["V_obs"].iloc[0]); vals, lab = scenarios.pdem_list(cfg)
    rows = []
    prof = None if mode_charge == "ferme" else F.profile
    for P in vals:
        Pd = loadprofile.pdem_series(F.idx, float(P), prof)
        for cas, pf, qi in (("hydro_seul", np.zeros(n), F.Q_in_h), ("hybride", F.fpv["P_fpv"].values, F.Q_in_h + F.Q_evap_saved_h)):
            sims = {"RBC": run_rbc(pf, Pd, qi, V0, F.res, F.plant, cfg), "DP": run_dp_hourly(pf, Pd, qi, V0, F.res, F.plant, cfg)}
            for strat, o in sims.items():
                k = kpi(o, F.res, cfg); v = validate(o, F.res, F.plant)
                rows.append(dict(P_dem_MW=P / 1e6, label=lab[P], cas=cas, strategie=strat, valid=v["ok"],
                                 jours_deficit=int((o["P_def"].reshape(-1, 24).sum(1) > 1e3).sum()),
                                 deficit_predit_DP_GWh=o["dp"]["cost_pred_GWh"] if strat == "DP" else np.nan, **k))
    R = pd.DataFrame(rows); Path(out_dir).mkdir(exist_ok=True); R.to_csv(Path(out_dir) / "kpi_dp_vs_rbc.csv", index=False)
    t = R.pivot_table(index="label", columns=["cas", "strategie"], values="E_def_GWh", sort=False).round(1)
    print("Déficit (GWh) :\n", t.to_string()); print("\nTests OK :", bool(R["valid"].all()))
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="resultats")
    ap.add_argument("--charge", default="ferme", choices=["ferme", "profil"]); a = ap.parse_args(); main(a.out, a.charge)
