"""Évaporation d'eau libre Penman FAO-56 (journalière) et température d'eau du lac (horaire)."""
import numpy as np
import pandas as pd


def daily_met(nasa):
    d = pd.DataFrame({
        "T": nasa["T2M"].resample("D").mean(),
        "RH": nasa["RH2M"].resample("D").mean(),
        "u2": nasa["WS2M"].resample("D").mean().clip(lower=0.1),
        "Rs_MJ": nasa["ALLSKY_SFC_SW_DWN"].resample("D").sum() * 3600 / 1e6})   # W/m2 horaire -> MJ/m2/j
    return d


def penman_open_water(dm, cfg):
    """E_open (mm/j). Port de bloc9_evaporation.m (Ra, Rso, Rnl, Rn, terme aérodynamique)."""
    p = cfg.evap
    T, RH, u2, Rs = dm["T"].values, dm["RH"].values, dm["u2"].values, dm["Rs_MJ"].values
    es = 0.6108 * np.exp(17.27 * T / (T + 237.3)); ea = es * RH / 100
    vpd = np.maximum(es - ea, 0)
    delta = 4098 * es / (T + 237.3) ** 2
    P_atm = 101.3 * ((293 - 0.0065 * p.z_site) / 293) ** 5.26
    gam = 0.000665 * P_atm
    phi = np.deg2rad(cfg.fpv.lat_deg)
    J = dm.index.dayofyear.values
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * J)
    ds = 0.409 * np.sin(2 * np.pi / 365 * J - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(ds), -1, 1))
    Ra = (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(phi) * np.sin(ds) + np.cos(phi) * np.cos(ds) * np.sin(ws))
    Rso = (0.75 + 2e-5 * p.z_site) * Ra
    rr = np.clip(Rs / np.maximum(Rso, 1e-9), 0.33, 1.0)
    Rnl = 4.903e-9 * (T + 273.16) ** 4 * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0.01))) * (1.35 * rr - 0.35)
    Rn = (1 - p.alpha_w) * Rs - Rnl
    lam = 2.45
    Ea = 6.43 * (1 + 0.536 * u2) * vpd / lam
    E = (delta * Rn / lam + gam * Ea) / (delta + gam)
    return pd.Series(np.maximum(E, 0), index=dm.index, name="E_open_mm_j")


def lake_water_temp(T_air, GHI, E_open_hourly_mm_j, p):
    """Lac peu profond (port de t_eau_lac.m, pas horaire)."""
    RHO_W, C_W, LAM = 1000.0, 4186.0, 2.45e6
    n = len(T_air); Te = np.empty(n); Te[0] = T_air[0]; dt = 3600.0
    for k in range(1, n):
        dT = (0.7 * GHI[k] - p.h_air * (Te[k - 1] - T_air[k]) - LAM * E_open_hourly_mm_j[k] / 86400.0) \
             * dt / (RHO_W * C_W * p.h_w)
        Te[k] = min(max(Te[k - 1] + dT, T_air[k] - 5), T_air[k] + 15)
    return Te
