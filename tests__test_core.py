import numpy as np
import pytest
from mwa.config import load_config
from mwa import forcings, data
from mwa.hydro import HydroPlant
from mwa.dispatch.rbc import run_rbc
from mwa.validation import validate


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def F(cfg):
    return forcings.build(cfg)


def test_qturb_fix_applied():
    h = data.load_hydro()
    assert h.loc["2025-03-11", "q_tur"] == pytest.approx(61.93)
    assert h["q_tur"].max() < 90


def test_reservoir_matches_bilan(cfg, F):
    b = data.load_bilan()
    err = np.abs(F.res.volume(b["niveau_relatif_m"].values) / 1e6 - b["volume_Mm3"].values)
    assert np.median(err) < 2.0 and err.max() < 25.0
    v = np.linspace(60e6, 600e6, 50)
    assert np.all(np.diff(F.res.level(v)) >= 0)          # z(V) croissante


def test_plant_nominal_unit(cfg):
    p = HydroPlant(cfg)
    z_for_111 = cfg.hydro.z_aval + 111.0 + cfg.hydro.k_loss * cfg.hydro.Q_nom_grp ** 2
    assert p.power(cfg.hydro.Q_nom_grp, z_for_111) == pytest.approx(13.2e6, rel=0.01)


def test_inverse_power_roundtrip(cfg):
    p = HydroPlant(cfg); p.build_table(205.5, 209.5)
    for q in (15, 30, 45, 60, 72):
        P = float(p.power(q, 207.5))
        assert float(p.q_for_power(P, 207.5)) == pytest.approx(q, abs=1.2)   # tolérance : bascule de groupes


def test_forcings_shape_and_fpv(cfg, F):
    assert len(F.idx) == 8016 and F.fpv.index.equals(F.idx)
    assert F.fpv["P_fpv"].max() <= cfg.fpv.P_AC + 1
    assert F.fpv.loc[F.idx.hour == 0, "P_fpv"].max() == 0.0
    kwh_kwc = F.fpv["P_fpv"].sum() / 1e3 / (cfg.fpv.P_DC / 1e3) * 365 / (len(F.idx) / 24)
    assert 1500 < kwh_kwc < 2100


def test_rbc_closure_and_validity(cfg, F):
    V0 = float(F.hydro_d["V_obs"].iloc[0])
    for P in (60e6, 90e6):
        o = run_rbc(F.fpv["P_fpv"].values, np.full(len(F.idx), P), F.Q_in_h, V0, F.res, F.plant, cfg)
        v = validate(o, F.res, F.plant)
        assert v["ok"], v
        assert abs(v["global_residual_Mm3"]) < 1e-6
        assert np.all(o["V"] <= F.res.V_max + 1)


def test_hybrid_never_worse_in_deficit(cfg, F):
    V0 = float(F.hydro_d["V_obs"].iloc[0]); P = np.full(len(F.idx), 75e6)
    ref = run_rbc(np.zeros(len(F.idx)), P, F.Q_in_h, V0, F.res, F.plant, cfg)
    hyb = run_rbc(F.fpv["P_fpv"].values, P, F.Q_in_h, V0, F.res, F.plant, cfg)
    assert hyb["P_def"].sum() <= ref["P_def"].sum()


def test_dp_realisation_matches_prediction(cfg, F):
    from mwa.dispatch.dp import run_dp_hourly
    n = len(F.idx); V0 = float(F.hydro_d["V_obs"].iloc[0]); Pd = np.full(n, 75e6)
    pf = F.fpv["P_fpv"].values
    d = run_dp_hourly(pf, Pd, F.Q_in_h, V0, F.res, F.plant, cfg)
    assert validate(d, F.res, F.plant)["ok"]
    real = d["P_def"].sum() * 3600 / 3.6e12
    assert abs(real - d["dp"]["cost_pred_GWh"]) < 0.05 * max(real, 1.0)      # écart horaire < 5 %
    rbc = run_rbc(pf, Pd, F.Q_in_h, V0, F.res, F.plant, cfg)
    assert real <= rbc["P_def"].sum() * 3600 / 3.6e12 * 1.03                 # DP au moins aussi bonne (à 3 % près)
