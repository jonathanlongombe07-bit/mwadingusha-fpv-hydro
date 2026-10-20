"""Courbes cote-volume-surface issues de la feuille BILAN et bilan de masse."""
import numpy as np
from scipy.interpolate import PchipInterpolator


def _pav(y, w=None):
    """Régression isotone croissante (pool-adjacent-violators)."""
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            wt = wts[-2] + wts[-1]; c = cnt[-2] + cnt[-1]
            vals[-2:] = [v]; wts[-2:] = [wt]; cnt[-2:] = [c]
    return np.repeat(vals, cnt)


def _monotone_curve(x, y):
    """Courbe y(x) croissante : moyenne des doublons, isotone, interpolation PCHIP."""
    order = np.argsort(x)
    x, y = np.asarray(x)[order], np.asarray(y)[order]
    ux, inv = np.unique(x, return_inverse=True)
    uy = np.bincount(inv, weights=y) / np.bincount(inv)
    uy = _pav(uy)
    # pchip exige des x strictement croissants ; y peut avoir des paliers
    return ux, uy, PchipInterpolator(ux, uy, extrapolate=False)


class Reservoir:
    """Relations V(z), S(V), z(V) en unités SI (m, m3, m2)."""

    def __init__(self, z_rel, vol_Mm3, surf_km2, V_min, V_max):
        zx, vy, self._V_of_z = _monotone_curve(z_rel, vol_Mm3)
        self.z_lo, self.z_hi = zx[0], zx[-1]
        self.v_lo, self.v_hi = vy[0], vy[-1]
        vx, sy, self._S_of_V = _monotone_curve(vol_Mm3, surf_km2)
        self._vgrid = vx
        # inverse z(V) : on rend V strictement croissant pour np.interp
        vv = vy + np.arange(len(vy)) * 1e-9
        self._vv, self._zz = vv, zx
        self.V_min, self.V_max = V_min, V_max

    # --- relations (entrées/sorties SI) ---
    def volume(self, z):
        z = np.clip(z, self.z_lo, self.z_hi)
        return self._V_of_z(z) * 1e6

    def level(self, V):
        return np.interp(np.asarray(V) / 1e6, self._vv, self._zz)

    def surface(self, V):
        v = np.clip(np.asarray(V) / 1e6, self._vgrid[0], self._vgrid[-1])
        return self._S_of_V(v) * 1e6

    @classmethod
    def from_table(cls, tab, cfg):
        r = cfg.reservoir
        return cls(tab["niveau_relatif_m"].values, tab["volume_Mm3"].values,
                   tab["surface_km2"].values, r.V_min_Mm3 * 1e6, r.V_max_Mm3 * 1e6)

    @classmethod
    def from_bilan(cls, bilan, cfg):
        r = cfg.reservoir
        return cls(bilan["niveau_relatif_m"].values, bilan["volume_Mm3"].values,
                   bilan["surface_km2"].values, r.V_min_Mm3 * 1e6, r.V_max_Mm3 * 1e6)


def inflow_from_balance(V, q_tur, q_dev, dt_s=86400.0, smooth_days=7):
    """Apport net : Q_in(t) = (V(t+1)-V(t))/dt + Q_tur + Q_dev, puis lissage centré.

    L'apport obtenu est NET de l'évaporation réelle du lac (elle est déjà dans la variation de volume).
    Les valeurs négatives sont conservées (physiquement possibles en saison sèche).
    """
    V = np.asarray(V, float)
    raw = np.empty_like(V)
    raw[:-1] = (V[1:] - V[:-1]) / dt_s + q_tur[:-1] + q_dev[:-1]
    raw[-1] = raw[-2]
    import pandas as pd
    sm = pd.Series(raw).rolling(smooth_days, center=True, min_periods=1).mean().values
    return raw, sm
