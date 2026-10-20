"""Production hydroélectrique NON LINÉAIRE : P = rho g eta(q) Q H_net(z, q).

- H_brute = z_rel - z_aval (variable avec le niveau du lac)
- pertes quadratiques par groupe, rendement paramétrique par groupe, nombre de groupes en marche
- inversion P -> Q par table (H, Q) monotone
"""
import numpy as np


class HydroPlant:
    def __init__(self, cfg):
        h = cfg.hydro
        self.h = h
        self.q_nom = h.Q_nom_grp
        self.n = h.n_groupes
        self.rho_g = h.rho * h.g
        self.Q_min = h.Q_tur_min
        self.Q_max = h.Q_tur_max_op
        self.P_cap = h.P_disponible
        self._tab = None

    # -- modèle direct (vectorisé) --
    def n_on(self, Q):
        return np.clip(np.ceil(np.asarray(Q, float) / (1.10 * self.q_nom)), 1, self.n)

    def eta(self, q_unit):
        x = np.asarray(q_unit) / self.q_nom
        return np.maximum(self.h.eta_nom * (1 - self.h.eta_c * (x - 1) ** 2), self.h.eta_min)

    def h_net(self, z_rel, Q):
        q_u = np.asarray(Q, float) / self.n_on(np.maximum(Q, 1e-9))
        return (np.asarray(z_rel) - self.h.z_aval) - self.h.k_loss * q_u ** 2

    def power(self, Q, z_rel):
        """Puissance (W) pour débit Q (m3/s) et niveau z_rel (m), sans plafonnement P_disponible."""
        Q = np.asarray(Q, float)
        q_u = Q / self.n_on(np.maximum(Q, 1e-9))
        Hn = (np.asarray(z_rel) - self.h.z_aval) - self.h.k_loss * q_u ** 2
        P = self.rho_g * self.eta(q_u) * Q * Hn
        return np.where(Q > 0, np.maximum(P, 0.0), 0.0)

    # -- inversion par table --
    def build_table(self, z_lo, z_hi, nz=40, nq=500):
        self._zg = np.linspace(z_lo, z_hi, nz)
        self._qg = np.linspace(0.0, self.Q_max, nq)
        P = np.array([self.power(self._qg, z) for z in self._zg])
        self._Pg = np.maximum.accumulate(P, axis=1)      # enveloppe monotone
        self._Pg = self._Pg + np.arange(nq) * 1e-6        # strictement croissante pour interp

    def q_for_power(self, P_target, z_rel):
        """Débit minimal fournissant P_target à ce niveau (borné à Q_max)."""
        z = float(np.clip(z_rel, self._zg[0], self._zg[-1]))
        i = int(np.clip(np.searchsorted(self._zg, z) - 1, 0, len(self._zg) - 2))
        w = (z - self._zg[i]) / (self._zg[i + 1] - self._zg[i])
        q0 = np.interp(P_target, self._Pg[i], self._qg)
        q1 = np.interp(P_target, self._Pg[i + 1], self._qg)
        return (1 - w) * q0 + w * q1
