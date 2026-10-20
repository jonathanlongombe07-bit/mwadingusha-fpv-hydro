"""12 scénarios hydro-climatiques (port de bloc19b_construction_scenarios.m).

Grille 2 saisons x 2 niveaux hydro x 3 conditions PV = 12 scénarios.
- saison : humide nov-avr / sèche mai-oct (mois du jour)
- niveau hydro : volume OBSERVÉ du jour > médiane (MaxHydric) ou <= médiane (MinHydric)
- condition PV : tercile de la puissance FPV moyenne journalière : > Q67 Sunny ; [Q33, Q67] Cloudy ; < Q33 Rainy
Différence avec le MATLAB (justifiée) : les scénarios sont des ENSEMBLES DE JOURS d'une simulation CONTIGUË sur l'année
(les indicateurs sont agrégés sur les jours du scénario), et non des jours dispersés rejoués séparément ; V(t) reste donc physique
et V_min n'est plus « indicatif ». Seuil de viabilité : 15 jours.
"""
import numpy as np
import pandas as pd

SAISONS = ("Wet", "Dry"); HYDROS = ("MaxHydric", "MinHydric"); PVS = ("Sunny", "Cloudy", "Rainy")


def build(F, seuil_min=15):
    h = F.hydro_d
    d = pd.DataFrame({"V_obs_Mm3": h["V_obs"] / 1e6})
    d["P_fpv_MW"] = (F.fpv["P_fpv"].groupby(F.idx.normalize()).mean() / 1e6).reindex(d.index)
    d = d[d["V_obs_Mm3"] > 0]
    wet = d.index.month.isin([11, 12, 1, 2, 3, 4])
    vmed = d["V_obs_Mm3"].median()
    q33, q67 = d["P_fpv_MW"].quantile([0.33, 0.67])
    masks = {}; rows = []; k = 0
    for s, ms in zip(SAISONS, (wet, ~wet)):
        for hy, mh in zip(HYDROS, (d["V_obs_Mm3"] > vmed, d["V_obs_Mm3"] <= vmed)):
            for pv, mp in zip(PVS, (d["P_fpv_MW"] > q67, (d["P_fpv_MW"] >= q33) & (d["P_fpv_MW"] <= q67), d["P_fpv_MW"] < q33)):
                k += 1; m = ms & mh & mp
                masks[k] = m.values
                rows.append(dict(id=k, nom=f"Scénario {k} : {s}, {hy}, {pv}", saison=s, hydro=hy, pv=pv, n_jours=int(m.sum()),
                                 viable=bool(m.sum() >= seuil_min)))
    info = dict(V_median_Mm3=float(vmed), P_Q33_MW=float(q33), P_Q67_MW=float(q67), n_jours=len(d), index=d.index)
    return pd.DataFrame(rows), masks, info
