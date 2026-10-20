"""Profil de charge normalisé (type de jour x heure) construit sur les jours 2024 les mieux couverts.

PROD SNEL = production mesurée, pas demande : on n'en retient que la FORME horaire, mise à l'échelle
par la consigne (ferme 60 MW ou surcharge). Les valeurs aberrantes (>80 MW, pics isolés) sont exclues.
"""
import numpy as np
import pandas as pd

BAD_FLAGS = ("gt_80MW", "pic_isole", "negatif", "texte_non_numerique")


def select_good_days(c, mois=(1, 8), min_hours=22):
    c = c.copy()
    c["date"] = c["date_sheet"].dt.normalize()
    c = c[c["statut_jour"] == "renseigne"]
    c = c[(c["date"].dt.month >= mois[0]) & (c["date"].dt.month <= mois[1])]
    bad = c["qc_flag"].fillna("").apply(lambda s: any(b in s for b in BAD_FLAGS))
    c = c[~bad & c["value"].notna()]
    n = c.groupby("date")["value"].count()
    ok = n[n >= min_hours].index
    return c[c["date"].isin(ok)]


def build_profile(c, mois=(1, 8)):
    g = select_good_days(c, mois)
    g = g.assign(hour=g["timestamp_proposed"].dt.hour,
                 weekend=g["timestamp_proposed"].dt.dayofweek >= 5)
    prof = g.groupby(["weekend", "hour"])["value"].mean().unstack(0)
    # normalisation : moyenne pondérée semaine (5/7) / week-end (2/7) = 1
    m = (prof[False] * 5 + prof[True] * 2).mean() / 7
    prof = prof / m
    info = {"jours_retenus": int(g["date"].nunique()), "mois": mois,
            "moyenne_MW_brute": float(g["value"].mean()),
            "ecart_type_forme": float(prof.values.std())}
    return prof, info


def pdem_series(index, P_ref_W, prof=None):
    """Consigne horaire (W). prof=None -> consigne ferme constante."""
    if prof is None:
        return np.full(len(index), float(P_ref_W))
    wk = (index.dayofweek >= 5)
    return P_ref_W * np.where(wk, prof[True].reindex(index.hour).values, prof[False].reindex(index.hour).values)
