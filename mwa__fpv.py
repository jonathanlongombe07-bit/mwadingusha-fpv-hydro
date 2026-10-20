"""PV flottant HORAIRE : transposition Hay-Davies (pvlib), T_cellule refroidie par l'eau, écrêtage onduleur."""
import numpy as np
import pandas as pd
import pvlib

from .evaporation import daily_met, penman_open_water, lake_water_temp


def solar_geometry(index_lst, cfg):
    """index_lst : horodatages en heure SOLAIRE locale (début d'heure). Instant = milieu d'heure."""
    mid = index_lst + pd.Timedelta(minutes=30)
    utc = (mid - pd.Timedelta(hours=cfg.fpv.lon_deg / 15.0)).tz_localize("UTC")
    sp = pvlib.solarposition.get_solarposition(utc, cfg.fpv.lat_deg, cfg.fpv.lon_deg)
    extra = pvlib.irradiance.get_extra_radiation(utc)
    return sp, extra.values


def simulate_fpv(nasa, cfg, N_mod=None):
    p = cfg.fpv
    N = p.N_mod if N_mod is None else N_mod
    idx = nasa.index
    sp, extra = solar_geometry(idx, cfg)
    zen = sp["apparent_zenith"].values; saz = sp["azimuth"].values
    ghi = nasa["ALLSKY_SFC_SW_DWN"].values.copy()
    dhi = np.minimum(nasa["ALLSKY_SFC_SW_DIFF"].values, ghi)        # corrige diffus > global (levers/couchers)
    cz = np.cos(np.deg2rad(zen))
    dni = np.where(zen < 88, (ghi - dhi) / np.maximum(cz, 0.087), 0.0)
    dni = np.clip(dni, 0, np.maximum(extra, 0))
    poa = pvlib.irradiance.get_total_irradiance(
        p.beta_deg, p.azimut_deg, zen, saz, dni, ghi, dhi, dni_extra=extra,
        model="haydavies", albedo=p.rho_sol)["poa_global"]
    poa = np.clip(np.nan_to_num(np.asarray(poa, float)), 0, None)

    dm = daily_met(nasa); E_open = penman_open_water(dm, cfg)
    E_h = E_open.reindex(idx.normalize()).values
    T_air = nasa["T2M"].values; v = np.maximum(nasa["WS2M"].values, 0.1)
    T_eau = lake_water_temp(T_air, ghi, E_h, cfg.evap)

    h_conv = p.U0 + p.U1 * v
    T_cell = T_air + (poa * (p.alpha_abs - p.eta_STC) + p.h_eau * (T_eau - T_air)) / (h_conv + p.h_eau)
    T_cell = np.clip(T_cell, T_air - 8, T_air + 40)
    f_temp = 1 + p.gamma * (T_cell - 25)
    f_low = np.where(poa > 1, 1 + p.alpha_low * np.log(np.maximum(poa, 1) / 1000), 0.0)
    f_temp = np.where(poa > 1, f_temp, 0.0)

    A = N * p.A_mod
    P_dc = A * poa * p.eta_STC * f_temp * f_low * p.eta_soil * p.eta_mism * p.eta_DC
    P_ac_brut = P_dc * p.eta_inv * p.eta_AC * p.eta_avail
    P_ac_max = p.P_AC * N / p.N_mod
    P_ac = np.minimum(np.maximum(P_ac_brut, 0), P_ac_max)
    out = pd.DataFrame({"G_POA": poa, "T_cell": T_cell, "T_eau": T_eau, "P_dc": P_dc,
                        "P_fpv": P_ac, "P_ecrete": np.maximum(P_ac_brut - P_ac_max, 0)}, index=idx)
    return out, E_open, A * p.kmarge   # A_occupee (m2)
