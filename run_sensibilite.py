"""Sensibilités §11 : (a) taux de couverture FPV, (b) indices de Sobol (S1, ST) sur 4 facteurs.

(a) f_cov = surface occupée par le FPV / surface du lac au volume plancher V_min. P_FPV et l'évaporation évitée sont PROPORTIONNELS au
    nombre de modules (plafond AC proportionnel) -> mise à l'échelle exacte, sans re-simuler pvlib.
(b) Facteurs (le cahier donne des plages pour un modèle journalier ; ADAPTATION HORAIRE 2025, signalée) :
      x1 multiplicateur d'apport Q_in in [0,5 ; 1,5]       (cahier : sigma_Q_in in [50,150] m3/s, non transposable)
      x2 biais GHI in [-0,05 ; +0,05] (GHI, DNI et DHI multipliés par 1+biais)   (cahier : identique)
      x3 h_eau in [50 ; 150] W/m2K                          (cahier : identique)
      x4 V0 in [0,8 ; 1,2] x V0 observé (118,6 Mm3)        (cahier : [500 ; 700] Mm3, incompatible avec V0 = 118,6 Mm3 le 1er janv. 2025)
    Sortie : déficit annuel (GWh), hybride, consigne 75 MW, RBC.  Estimateurs : Saltelli 2010 (S1), Jansen (ST), N de base = 64 (indicatif).
Usage : python run_sensibilite.py [--N 64] [--procs 2]
"""
import argparse
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import qmc

from mwa.config import load_config, replace
from mwa import forcings, loadprofile, data
from mwa.fpv import simulate_fpv
from mwa.dispatch.rbc import run_rbc
from mwa.dispatch.dp import run_dp_hourly

_G = {}


def _init():
    cfg = load_config(); F = forcings.build(cfg)
    _G.update(cfg=cfg, F=F, nasa=data.load_nasa().loc[F.idx[0]:F.idx[-1]])


def _coverage(f):
    cfg, F = _G["cfg"], _G["F"]; n = len(F.idx); V0 = float(F.hydro_d["V_obs"].iloc[0])
    S_min = float(F.res.surface(F.res.V_min)); N0 = cfg.fpv.N_mod
    f0 = F.A_fpv_occ / S_min
    sc = f / f0 if f0 > 0 else 0.0
    pf = F.fpv["P_fpv"].values * sc
    saved = F.Q_evap_saved_h * sc
    rows = []
    for P in (60.0, 75.0):
        Pd = loadprofile.pdem_series(F.idx, P * 1e6, None)
        for strat in ("RBC", "DP"):
            qi = F.Q_in_h + saved
            o = run_rbc(pf, Pd, qi, V0, F.res, F.plant, cfg) if strat == "RBC" else run_dp_hourly(pf, Pd, qi, V0, F.res, F.plant, cfg)
            rows.append(dict(f_cov_pct=100 * f, N_mod=N0 * sc, P_DC_MWp=cfg.fpv.P_DC * sc / 1e6, P_dem_MW=P, strategie=strat,
                             E_def_GWh=o["P_def"].sum() * 3600 / 3.6e12, E_fpv_GWh=pf.sum() * 3600 / 3.6e12, E_curt_GWh=o["P_curt"].sum() * 3600 / 3.6e12,
                             V_min_Mm3=o["V"].min() / 1e6, evap_evitee_Mm3=saved.sum() * 3600 / 1e6))
    return rows


def _sobol_eval(x):
    cfg, F, nasa = _G["cfg"], _G["F"], _G["nasa"].copy()
    q_mult, bias, h_eau, v0m = x
    for c in ("ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DIFF", "CLRSKY_SFC_SW_DWN"):
        nasa[c] = nasa[c] * (1 + bias)
    c2 = replace(cfg, fpv={"h_eau": float(h_eau)})
    fpv, _, _ = simulate_fpv(nasa, c2)
    pf = fpv["P_fpv"].values; n = len(pf)
    Pd = loadprofile.pdem_series(F.idx, 75e6, None)
    qi = F.Q_in_h * q_mult + F.Q_evap_saved_h
    o = run_rbc(pf, Pd, qi, float(F.hydro_d["V_obs"].iloc[0]) * v0m, F.res, F.plant, cfg)
    return o["P_def"].sum() * 3600 / 3.6e12


def main(N=64, procs=2, out="resultats"):
    outd = Path(out); outd.mkdir(exist_ok=True)
    with Pool(procs, initializer=_init) as pool:
        fl = [0.0, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0, 30.0]
        rows = [r for rr in pool.map(_coverage, [f / 100 for f in fl]) for r in rr]
        pd.DataFrame(rows).to_csv(outd / "sensibilite_couverture.csv", index=False)
        print(pd.DataFrame(rows).round(2).to_string(index=False), flush=True)
        lo = np.array([0.5, -0.05, 50.0, 0.8]); hi = np.array([1.5, 0.05, 150.0, 1.2]); D = 4
        AB = qmc.Sobol(2 * D, scramble=True, seed=1).random(N)
        A, B = lo + AB[:, :D] * (hi - lo), lo + AB[:, D:] * (hi - lo)
        ABi = []
        for i in range(D):
            C = A.copy(); C[:, i] = B[:, i]; ABi.append(C)
        X = np.vstack([A, B] + ABi)
        y = np.array(pool.map(_sobol_eval, list(X), chunksize=4))
    yA, yB = y[:N], y[N:2 * N]; yC = [y[(2 + i) * N:(3 + i) * N] for i in range(D)]
    V = np.var(np.concatenate([yA, yB]), ddof=1)
    names = ["Q_in x", "biais GHI", "h_eau", "V0 x"]
    S1 = [np.mean(yB * (yC[i] - yA)) / V for i in range(D)]
    ST = [0.5 * np.mean((yA - yC[i]) ** 2) / V for i in range(D)]
    df = pd.DataFrame(dict(facteur=names, S1=S1, ST=ST)); df.attrs = {}
    df.to_csv(outd / "sobol_indices.csv", index=False)
    print(f"\nSobol (N={N}, {len(X)} évaluations) ; variance de la sortie {V:.2f} GWh² ; moyenne {y.mean():.1f} GWh\n", df.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--N", type=int, default=64); ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args(); main(a.N, a.procs)
