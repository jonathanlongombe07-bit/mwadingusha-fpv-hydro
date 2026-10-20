"""Étape 1-2 : forçages 2025, rejeu historique, RBC hydro seul vs hybride, surcharges, stress.

Usage : python run_step1.py [--out resultats]
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from mwa.config import load_config
from mwa import forcings, loadprofile, scenarios
from mwa.dispatch.rbc import run_rbc
from mwa.validation import validate
from mwa.indicators import kpi


def main(out_dir="resultats", mode_charge="ferme"):
    cfg = load_config(); F = forcings.build(cfg)
    out = Path(out_dir); out.mkdir(exist_ok=True)
    print(f"Période {F.idx[0]} -> {F.idx[-1]}  ({len(F.idx)} h)")
    print(f"Profil de charge : {F.profile_info}")

    # --- rejeu historique (turbinage observé) : écart de volume ---
    hd = F.hydro_d
    V = hd["V_obs"].values.copy(); Vsim = [V[0]]
    for t in range(len(hd) - 1):
        Vsim.append(Vsim[-1] + 86400 * (hd["q_in"].iloc[t] - hd["q_tur"].iloc[t] - hd["q_dev"].iloc[t]))
    rmse = float(np.sqrt(np.mean((np.array(Vsim) - V) ** 2)) / 1e6)
    print(f"Rejeu historique : RMSE volume = {rmse:.2f} Mm3 (apport lissé {cfg.reservoir.lissage_apport_jours} j)")
    P_obs = F.plant.power(hd["q_tur"].values, hd["z_rel"].values) / 1e6
    print("Puissance hydro implicite 2025 (MW) par mois :\n",
          pd.Series(P_obs, index=hd.index).groupby(hd.index.month).mean().round(1).to_dict())

    prof = None if mode_charge == "ferme" else F.profile
    vals, lab = scenarios.pdem_list(cfg); st = scenarios.stress_cases(cfg)
    V0 = float(hd["V_obs"].iloc[0]); rows = []; monthly = []
    for sid, s in st.items():
        for pd_W in vals:
            Pdem = loadprofile.pdem_series(F.idx, pd_W, prof)
            Qin = F.Q_in_h * s["Q_in_factor"]
            ref = run_rbc(np.zeros(len(F.idx)), Pdem, Qin, V0, F.res, F.plant, cfg, Q_tur_max=s["Q_tur_max"])
            hyb = run_rbc(F.fpv["P_fpv"].values, Pdem, Qin + F.Q_evap_saved_h, V0, F.res, F.plant, cfg,
                          Q_tur_max=s["Q_tur_max"])
            for nom, o in (("hydro_seul", ref), ("hybride", hyb)):
                v = validate(o, F.res, F.plant)
                k = kpi(o, F.res, cfg, None if nom == "hydro_seul" else ref)
                rows.append(dict(stress=sid, P_dem_MW=pd_W / 1e6, label=lab[pd_W], cas=nom, valid=v["ok"],
                                 res_masse_m3=v["mass_residual_m3"], **k))
            m = pd.DataFrame({"def_ref": ref["P_def"], "def_hyb": hyb["P_def"]}, index=F.idx)
            g = (m.groupby(m.index.month).sum() * 3600 / 3.6e12).round(3)
            g.insert(0, "P_dem_MW", pd_W / 1e6); g.insert(0, "stress", sid); monthly.append(g.reset_index(names="mois"))
    R = pd.DataFrame(rows); R.to_csv(out / "kpi_rbc.csv", index=False)
    pd.concat(monthly).to_csv(out / "deficit_mensuel.csv", index=False)
    F.fpv.assign(Q_in=F.Q_in_h).to_csv(out / "forcages_horaires.csv")
    pd.set_option("display.width", 200)
    b = R[(R.stress == "base") & (R.cas == "hybride")][["label", "E_dem_GWh", "E_def_GWh", "deficit_evite_GWh", "R",
                                                          "eau_economisee_Mm3", "V_min_Mm3", "valid"]]
    print("\n=== Base, hybride vs hydro seul ===\n", b.round(3).to_string(index=False))
    print("\nTous les tests de cohérence OK :", bool(R["valid"].all()))
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="resultats")
    ap.add_argument("--charge", default="ferme", choices=["ferme", "profil"]); a = ap.parse_args()
    main(a.out, a.charge)
