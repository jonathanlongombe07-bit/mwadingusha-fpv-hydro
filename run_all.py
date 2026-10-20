"""Chaîne complète : 4 stratégies (RBC, DP, MILP, SMPC) x {hydro seul, hybride} x consignes x stress, indicateurs globaux et par scénario,
6 tests formels, résultats journaliers.

Usage : python run_all.py [--out resultats] [--procs 2] [--quick]
Durée : ~25 min avec 2 processus (MILP ~35 s par run).  --quick : consignes 60 et 75 MW, base seulement.
"""
import argparse, time, json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

from mwa.config import load_config
from mwa import forcings, loadprofile, scenarios, hydroclimat
from mwa.dispatch.rbc import run_rbc
from mwa.dispatch.dp import run_dp_hourly
from mwa.dispatch.milp import run_milp
from mwa.dispatch.smpc import run_smpc
from mwa.indicators import kpi_ext, daily_frame
from mwa.validation import formal_tests

_G = {}


def _init():
    cfg = load_config(); F = forcings.build(cfg)
    sc, masks, info = hydroclimat.build(F)
    _G.update(cfg=cfg, F=F, sc=sc, masks=masks, info=info)


def _scenario_rows(o, keyinfo):
    F = _G["F"]; sc = _G["sc"]; masks = _G["masks"]; GWh = 3600.0 / 3.6e12
    df = daily_frame(o, F.idx); dsel = _G["info"]["index"]
    df = df.reindex(dsel)
    rows = []
    for _, r in sc.iterrows():
        m = masks[r["id"]]
        if r["n_jours"] == 0: continue
        sub = df[m]
        e_def = sub["P_def_MW"].sum() * 24 / 1e3; e_dem = sub["P_dem_MW"].sum() * 24 / 1e3
        rows.append(dict(**keyinfo, scenario_id=int(r["id"]), scenario=r["nom"], n_jours=int(r["n_jours"]), viable=bool(r["viable"]),
                         E_def_GWh=e_def, E_dem_GWh=e_dem, R=1 - e_def / max(e_dem, 1e-9),
                         jours_deficit_pct=100 * float((sub["P_def_MW"] > 1e-3).mean()), P_FPV_moy_MW=sub["P_FPV_MW"].mean(),
                         P_H_moy_MW=sub["P_H_MW"].mean(), Q_tur_moy=sub["Q_tur_m3s"].mean(), V_min_fin_jour_Mm3=sub["V_fin_Mm3"].min()))
    return rows


def _task(args):
    cas, P_MW, stress, charge = args
    cfg, F = _G["cfg"], _G["F"]; n = len(F.idx)
    st = scenarios.stress_cases(cfg)[stress]
    prof = None if charge == "ferme" else F.profile
    Pd = loadprofile.pdem_series(F.idx, P_MW * 1e6, prof)
    V0 = float(F.hydro_d["V_obs"].iloc[0])
    qin = F.Q_in_h * st["Q_in_factor"]
    if cas == "hybride":
        pf = F.fpv["P_fpv"].values; qi = qin + F.Q_evap_saved_h
    else:
        pf = np.zeros(n); qi = qin
    Qmax = st["Q_tur_max"]
    res, plant = F.res, F.plant
    out = {}; tim = {}
    t = time.time(); out["RBC"] = run_rbc(pf, Pd, qi, V0, res, plant, cfg, Q_tur_max=Qmax); tim["RBC"] = time.time() - t
    t = time.time(); out["DP"] = run_dp_hourly(pf, Pd, qi, V0, res, plant, cfg, Q_tur_max=Qmax); tim["DP"] = time.time() - t
    t = time.time()
    out["MILP"] = run_milp(pf, Pd, qi, V0, res, plant, cfg, Q_tur_max=Qmax, n_iter=1, window_days=7, V_ref_traj=out["DP"]["V"],
                           V_end_ref=out["DP"]["V"], time_limit=60, gap=1e-3); tim["MILP"] = time.time() - t
    t = time.time(); out["SMPC"] = run_smpc(pf, Pd, qi, V0, res, plant, cfg, Q_tur_max=Qmax); tim["SMPC"] = time.time() - t
    rows, srows, trows, drows = [], [], [], []
    for strat, o in out.items():
        key = dict(charge=charge, stress=stress, P_dem_MW=P_MW, cas=cas, strategie=strat)
        k = kpi_ext(o, res, cfg, Q_in_tot_Mm3=float(qi.sum() * 3600 / 1e6))
        rows.append(dict(**key, temps_s=tim[strat], **k))
        srows += _scenario_rows(o, key)
        for name, (ok, val) in formal_tests(o, res, plant, cfg).items():
            trows.append(dict(**key, test=name, PASS=bool(ok), valeur=val))
        if stress == "base" and charge == "ferme" and P_MW in (60.0, 75.0):
            d = daily_frame(o, F.idx, T_cell=F.fpv["T_cell"].values); d = d.assign(**key)
            drows.append(d.reset_index())
    return rows, srows, trows, drows


def main(out_dir="resultats", procs=2, quick=False):
    cfg = load_config(); outd = Path(out_dir); outd.mkdir(exist_ok=True)
    _init()
    sc = _G["sc"]; sc.to_csv(outd / "scenarios_12_effectifs.csv", index=False)
    print(json.dumps({k: v for k, v in _G["info"].items() if k != "index"}), flush=True)
    print(sc[["id", "nom", "n_jours", "viable"]].to_string(index=False), flush=True)
    vals, lab = scenarios.pdem_list(cfg)
    std = [60.0, 65.0, 70.0, 75.0] if not quick else [60.0, 75.0]
    over = [78.0, 84.0, 90.0] if not quick else []
    tasks = []
    for cas in ("hydro_seul", "hybride"):
        for stress in (("base", "S1", "S2", "S3") if not quick else ("base",)):
            for P in std: tasks.append((cas, P, stress, "ferme"))
        for P in over: tasks.append((cas, P, "base", "ferme"))
    tasks += [("hybride", 60.0, "base", "profil"), ("hybride", 75.0, "base", "profil"),
              ("hydro_seul", 60.0, "base", "profil"), ("hydro_seul", 75.0, "base", "profil")] if not quick else []
    print(f"{len(tasks)} combinaisons x 4 stratégies", flush=True)
    R, S, T, D = [], [], [], []
    t0 = time.time()
    with Pool(procs, initializer=_init) as pool:
        for i, (r, s, t, d) in enumerate(pool.imap_unordered(_task, tasks), 1):
            R += r; S += s; T += t; D += d
            print(f"  [{i}/{len(tasks)}] {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(R).to_csv(outd / "matrice_kpi_4strategies.csv", index=False)
    pd.DataFrame(S).to_csv(outd / "kpi_par_scenario.csv", index=False)
    pd.DataFrame(T).to_csv(outd / "validation_6tests.csv", index=False)
    if D: pd.concat(D).to_csv(outd / "journalier_base.csv", index=False)
    print("terminé", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="resultats"); ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--quick", action="store_true"); a = ap.parse_args(); main(a.out, a.procs, a.quick)
