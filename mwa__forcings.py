"""Assemblage des forçages horaires 2025 (hydro journalier -> horaire, FPV horaire, charge 2024)."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from . import data, loadprofile
from .reservoir import Reservoir, inflow_from_balance
from .hydro import HydroPlant
from .fpv import simulate_fpv


@dataclass
class Forcings:
    idx: pd.DatetimeIndex
    hydro_d: pd.DataFrame      # journalier : z_rel, q_tur, q_dev, V_obs, q_in_brut, q_in
    res: Reservoir
    plant: HydroPlant
    fpv: pd.DataFrame          # horaire
    E_open: pd.Series          # journalier mm/j
    A_fpv_occ: float
    Q_in_h: np.ndarray         # horaire, net de l'évaporation historique
    Q_evap_saved_h: np.ndarray  # horaire, gain d'évaporation dû à la couverture FPV
    profile: pd.DataFrame
    profile_info: dict


def build(cfg, data_dir=None, fpv_N_mod=None):
    hyd = data.load_hydro(data_dir); bil = data.load_bilan(data_dir)
    t0, t1 = pd.Timestamp(cfg.periode.debut), pd.Timestamp(cfg.periode.fin)
    hyd = hyd.loc[t0:t1]
    res = Reservoir.from_table(data.load_courbes(data_dir, cfg.reservoir.courbes_csv), cfg)
    hyd["V_obs"] = res.volume(hyd["z_rel"].values)
    raw, sm = inflow_from_balance(hyd["V_obs"].values, hyd["q_tur"].values, hyd["q_dev"].values,
                                  smooth_days=cfg.reservoir.lissage_apport_jours)
    hyd["q_in_brut"], hyd["q_in"] = raw, sm

    idx = pd.date_range(t0, t1 + pd.Timedelta(hours=23), freq="h")
    nasa = data.load_nasa(data_dir).loc[idx[0]:idx[-1]]
    assert len(nasa) == len(idx), "NASA POWER incomplet sur la période"
    fpv, E_open, A_occ = simulate_fpv(nasa, cfg, N_mod=fpv_N_mod)

    day = idx.normalize()
    q_in_h = hyd["q_in"].reindex(day).values
    S_d = res.surface(hyd["V_obs"].values)
    S_h = pd.Series(S_d, index=hyd.index).reindex(day).values
    E_h = E_open.reindex(day).values
    saved = cfg.evap.r_evap * np.minimum(A_occ, S_h) * E_h / 1000.0 / 86400.0   # m3/s

    plant = HydroPlant(cfg); plant.build_table(float(hyd["z_rel"].min()) - 0.5, float(hyd["z_rel"].max()) + 0.5)
    prof, info = loadprofile.build_profile(data.load_charges(data_dir), tuple(cfg.periode.fenetre_charge_2024))
    return Forcings(idx, hyd, res, plant, fpv, E_open, A_occ, q_in_h, saved, prof, info)
