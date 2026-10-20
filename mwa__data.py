"""Chargeurs des CSV produits par l'extraction (dossier data/)."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
QTUR_FIX = {"2025-03-11": 61.93}   # 621.93 dans DONNEES = faute de frappe ; BILAN indique 61.93 (décision utilisateur)


def load_hydro(data_dir=None):
    d = Path(data_dir) if data_dir else ROOT / "data"
    h = pd.read_csv(d / "hydrologie_2025.csv", parse_dates=["date"])
    h = h[h["renseigne"] == True].copy()   # noqa: E712
    for k, v in QTUR_FIX.items():
        h.loc[h["date"] == k, "msha_qturb"] = v
    h = h.set_index("date").sort_index()
    out = pd.DataFrame({
        "z_rel": h["msha_niveau_relatif"], "z_cote": h["msha_niveau_cote"],
        "q_tur": h["msha_qturb"].fillna(0.0), "q_dev": h["msha_qdeverse"].fillna(0.0),
        "apport_fichier": h["msha_apport_1j"], "chute_obs": h["msha_chute"]})
    return out


def load_bilan(data_dir=None):
    d = Path(data_dir) if data_dir else ROOT / "data"
    b = pd.read_csv(d / "annexe_niveau_surface_volume_2025.csv", parse_dates=["date"])
    return b.set_index("date").sort_index()


def load_courbes(data_dir=None, name="courbes_feuil16.csv"):
    d = Path(data_dir) if data_dir else ROOT / "data"
    return pd.read_csv(d / name)


def load_nasa(data_dir=None):
    d = Path(data_dir) if data_dir else ROOT / "data"
    n = pd.read_csv(d / "nasa_power_2025_hourly_wide.csv", parse_dates=["timestamp"])
    return n.set_index("timestamp").sort_index()


def load_charges(data_dir=None):
    d = Path(data_dir) if data_dir else ROOT / "data"
    c = pd.read_csv(d / "charges_mwandingusha_2024_hourly.csv", parse_dates=["date_sheet", "timestamp_proposed"])
    return c[~c["is_total"]].copy()
