"""Figures de qualité publication — port Python des scripts MATLAB de visualisation
(style_publication, save_figure, visualisation_part2/3, visualisation_surcharge, untitled5 = annexe C,
untitled6 = Spearman, visu_chap6_hydro / visu_chap6_dispatch).

NON PORTÉ ici (dépend de blocs pas encore portés) : MILP / SMPC (remplacés par RBC vs DP),
visualisation_part4 (12 scénarios hydro-climatiques), figures économiques (visu_chap6_economie :
prix et énergies en dur dans le MATLAB, aucune source -> à refaire quand le module économique existera).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BLEU, VERT, ORANGE, ROUGE, JAUNE, CYAN = "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#fcbf14", "#17becf"
MOIS = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]


def style_publication():
    """Équivalent de style_publication.m (Helvetica -> DejaVu/Arial selon dispo, 11 pt, traits 1.8)."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5, "legend.fontsize": 10,
        "lines.linewidth": 1.8, "lines.markersize": 7, "axes.linewidth": 1.0,
        "xtick.direction": "out", "ytick.direction": "out", "axes.grid": True, "grid.alpha": 0.3,
        "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "figure.dpi": 100, "axes.titleweight": "bold",
    })


def save_figure(fig, base, dpi=300):
    """PNG 300 dpi + PDF vectoriel (comme save_figure.m)."""
    base = Path(base); base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return base.with_suffix(".png")


def _daily(x, how="mean"):
    return getattr(x.groupby(x.index.normalize()), how)()


# ---------------------------------------------------------------------------------------------
# Partie 2 / annexe C : forçages, FPV, évaporation, réservoir
# ---------------------------------------------------------------------------------------------
def fig_forcages_meteo(nasa, F, d):
    fig, ax = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    n = nasa.loc[F.idx[0]:F.idx[-1]]
    for a, (col, lab, c, ttl) in zip(ax, [
            ("ALLSKY_SFC_SW_DWN", "GHI (W/m²)", JAUNE, "(a) Irradiance globale horizontale"),
            ("T2M", "T$_{air}$ (°C)", ROUGE, "(b) Température de l'air"),
            ("WS2M", "v (m/s)", BLEU, "(c) Vitesse du vent"),
            ("RH2M", "HR (%)", CYAN, "(d) Humidité relative")]):
        a.plot(_daily(n[col]), color=c, lw=0.9); a.set_ylabel(lab); a.set_title(ttl, fontweight="normal", loc="left")
    ax[-1].set_xlabel("Date")
    fig.suptitle("Forçages météorologiques NASA POWER 2025 — Mwadingusha (moyennes journalières)", fontweight="bold")
    return save_figure(fig, d / "fig_p2_01_forcages_meteo")


def fig_fpv(F, d):
    p = F.fpv["P_fpv"] / 1e6
    pj = _daily(p)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
    ax[0].plot(pj, color=ORANGE, lw=0.8, label="Moyenne journalière")
    ax[0].plot(pj.rolling(30, center=True, min_periods=5).mean(), color=ROUGE, lw=2.2, label="Moyenne mobile 30 j")
    ax[0].set(xlabel="Date", ylabel="P$_{FPV}$ (MW)"); ax[0].set_title("(a) Puissance FPV journalière", loc="left"); ax[0].legend()
    ax[1].hist(pj, bins=25, color=ORANGE, edgecolor="k"); ax[1].set(xlabel="P$_{FPV}$ moyenne journalière (MW)", ylabel="Jours")
    ax[1].set_title("(b) Distribution", loc="left")
    fig.suptitle("Production photovoltaïque flottante (écrêtage AC 75 MW)", fontweight="bold")
    return save_figure(fig, d / "fig_p2_02_puissance_fpv")


def fig_evaporation(F, d):
    E = F.E_open
    saved = pd.Series(F.Q_evap_saved_h, index=F.idx)
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax[0].plot(E.loc[F.idx[0]:F.idx[-1]], color=CYAN, lw=1.0); ax[0].set_ylabel("E$_{eau libre}$ (mm/j)")
    ax[0].set_title("(a) Évaporation eau libre (Penman FAO-56)", loc="left")
    ax[1].plot(_daily(saved), color=BLEU, lw=1.0); ax[1].set(ylabel="Débit évité (m³/s)", xlabel="Date")
    ax[1].set_title("(b) Évaporation évitée par la couverture FPV", loc="left")
    fig.suptitle("Évaporation sur le lac", fontweight="bold")
    return save_figure(fig, d / "fig_p2_03_evaporation")


def fig_cycle_annuel(F, d):
    e = (F.fpv["P_fpv"] / 1e9).groupby(F.idx.month).sum()          # GWh
    s = (pd.Series(F.Q_evap_saved_h, index=F.idx) * 3600 / 1e6).groupby(F.idx.month).sum()   # Mm3
    fig, ax = plt.subplots(figsize=(10, 4.8)); ax2 = ax.twinx()
    ax.bar(e.index - 0.2, e.values, 0.4, color=ORANGE, edgecolor="k", label="Énergie FPV (GWh)")
    ax2.bar(s.index + 0.2, s.values, 0.4, color=CYAN, edgecolor="k", label="Eau évaporation évitée (Mm³)")
    ax.set_xticks(e.index); ax.set_xticklabels([MOIS[m - 1] for m in e.index])
    ax.set_ylabel("Énergie FPV (GWh)"); ax2.set_ylabel("Évaporation évitée (Mm³)"); ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="upper right")
    ax.set_title("Cycle annuel — FPV vs évaporation évitée")
    return save_figure(fig, d / "fig_p2_04_cycle_annuel")


def fig_reservoir_hist(F, d, name="fig_p2_05_reservoir"):
    h = F.hydro_d
    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    ax[0].plot(h.index, h["V_obs"] / 1e6, color=BLEU); ax[0].axhline(F.res.V_min / 1e6, color="k", ls="--", lw=1.2)
    ax[0].text(h.index[3], F.res.V_min / 1e6 + 12, "V$_{min}$ (plancher d'exploitation)", fontsize=9)
    ax[0].set_ylabel("Volume (Mm³)"); ax[0].set_title("(a) Volume du réservoir (niveau observé + Feuil16)", loc="left")
    ax[1].plot(h.index, h["q_in"], color=BLEU, label="Q$_{in}$ reconstruit (lissé 7 j)"); ax[1].plot(h.index, h["q_tur"], color=VERT, label="Q$_{tur}$ SNEL")
    ax[1].set_ylabel("Débit (m³/s)"); ax[1].legend(); ax[1].set_title("(b) Débits entrant et turbiné", loc="left")
    ax[2].plot(h.index, h["q_dev"], color=ROUGE); ax[2].set(ylabel="Q$_{dév}$ (m³/s)", xlabel="Date"); ax[2].set_title("(c) Déversement", loc="left")
    fig.suptitle("Bilan hydrique du réservoir — 2025", fontweight="bold")
    return save_figure(fig, d / name)


def fig_annexeC_debits(F, d):
    h = F.hydro_d
    fig, ax = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    ax[0].plot(h.index, h["q_tur"], color=VERT, label="Q$_{tur}$"); ax[0].plot(h.index, h["q_dev"], color=ROUGE, label="Q$_{dév}$")
    ax[0].set_ylabel("Débit (m³/s)"); ax[0].legend(); ax[0].set_title("(a) Débits turbiné et déversé mesurés (SNEL)", loc="left")
    ax[1].plot(h.index, h["q_in_brut"], color="0.7", lw=0.8, label="brut"); ax[1].plot(h.index, h["q_in"], color=BLEU, label="lissé 7 j")
    ax[1].set(ylabel="Q$_{in}$ (m³/s)", xlabel="Date"); ax[1].legend(); ax[1].set_title("(b) Apport reconstruit Q$_{in}$ (net d'évaporation)", loc="left")
    return save_figure(fig, d / "annexe_C_debits_hydro")


def tables_mensuelles(nasa, F, d):
    """Équivalent des CSV annexe_C_*_mensuel.csv."""
    d = Path(d); h = F.hydro_d; i = pd.Index(h.index)
    hm = pd.DataFrame({"V_moy_Mm3": (h["V_obs"] / 1e6).groupby(i.month).mean(), "V_min_Mm3": (h["V_obs"] / 1e6).groupby(i.month).min(),
                       "V_max_Mm3": (h["V_obs"] / 1e6).groupby(i.month).max(), "Q_tur_moy": h["q_tur"].groupby(i.month).mean(),
                       "Q_dev_moy": h["q_dev"].groupby(i.month).mean()}); hm.index.name = "mois"
    hm.round(3).to_csv(d / "annexe_C_hydro_mensuel.csv")
    n = nasa.loc[F.idx[0]:F.idx[-1]]; m = n.index.month
    mm = pd.DataFrame({"GHI_moy_Wm2": n["ALLSKY_SFC_SW_DWN"].groupby(m).mean(), "Tair_moy_C": n["T2M"].groupby(m).mean(),
                       "RH_moy_pct": n["RH2M"].groupby(m).mean(), "v_moy_ms": n["WS2M"].groupby(m).mean(),
                       "Pluie_cum_mm": _daily(n["PRECTOTCORR"], "mean").groupby(lambda t: t.month).sum()}); mm.index.name = "mois"
    mm.round(3).to_csv(d / "annexe_C_meteo_mensuel.csv")
    f = F.fpv; fm = pd.DataFrame({"P_FPV_moy_MW": (f["P_fpv"] / 1e6).groupby(F.idx.month).mean(),
                                  "E_FPV_GWh": (f["P_fpv"] * 3600 / 3.6e12).groupby(F.idx.month).sum(),
                                  "Tcell_moy_C": f["T_cell"].groupby(F.idx.month).mean()}); fm.index.name = "mois"
    fm.round(3).to_csv(d / "annexe_C_fpv_mensuel.csv")


def fig_spearman(nasa, F, d):
    """Matrice de Spearman sur 11 variables journalières (équivalent untitled6.m, 2025)."""
    h = F.hydro_d; n = nasa.loc[F.idx[0]:F.idx[-1]]
    dj = pd.DataFrame({
        "V (Mm³)": h["V_obs"] / 1e6, "Q$_{tur}$ (m³/s)": h["q_tur"], "Q$_{in}$ (m³/s)": h["q_in"],
        "GHI (W/m²)": _daily(n["ALLSKY_SFC_SW_DWN"]), "T$_{air}$ (°C)": _daily(n["T2M"]), "HR (%)": _daily(n["RH2M"]),
        "v$_{vent}$ (m/s)": _daily(n["WS2M"]), "P$_{FPV}$ (MW)": _daily(F.fpv["P_fpv"] / 1e6),
        "T$_{cell}$ (°C)": _daily(F.fpv["T_cell"]), "E$_{open}$ (mm/j)": F.E_open, "Pluie (mm/j)": _daily(n["PRECTOTCORR"]),
    }).dropna()
    rho, p = stats.spearmanr(dj.values)
    pd.DataFrame(rho, index=dj.columns, columns=dj.columns).round(4).to_csv(d / "annexe_C_correlation_spearman.csv")
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1); ax.grid(False)
    k = len(dj.columns); ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(dj.columns, rotation=45, ha="right"); ax.set_yticklabels(dj.columns)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{rho[i, j]:.2f}" + ("*" if p[i, j] < 0.05 and i != j else ""), ha="center", va="center",
                    fontsize=8, color="w" if abs(rho[i, j]) > 0.6 else "k")
    fig.colorbar(im, ax=ax, label="ρ de Spearman", shrink=0.8)
    ax.set_title(f"Corrélation de Spearman (2025, {len(dj)} jours ; * p<0,05)")
    return save_figure(fig, d / "annexe_C_correlation_spearman")


# ---------------------------------------------------------------------------------------------
# Partie 3 / chapitre 6 : stratégies (RBC vs DP) — A = hydro seul, B = hybride
# ---------------------------------------------------------------------------------------------
def _daily_arr(x, idx, how="sum"):
    return getattr(pd.Series(x, index=idx).groupby(idx.normalize()), how)()


def fig_comparaison_globale(runs, d, P_label="60 (ref)"):
    """runs[(cas, strat)] = out ; barres déficit / eau turbinée / déversé / V final."""
    keys = [("hydro_seul", "RBC"), ("hydro_seul", "DP"), ("hybride", "RBC"), ("hybride", "DP")]
    lab = ["A RBC", "A DP", "B RBC", "B DP"]; col = [BLEU, VERT, BLEU, VERT]
    G = lambda o: o["dt"] / 3.6e12
    vals = {
        "(a) Déficit (GWh)": [runs[k]["P_def"].sum() * G(runs[k]) for k in keys],
        "(b) Eau turbinée (Mm³)": [runs[k]["Q_tur"].sum() * 3600 / 1e6 for k in keys],
        "(c) Déversement (Mm³)": [runs[k]["Q_spill"].sum() * 3600 / 1e6 for k in keys],
        "(d) Volume final (Mm³)": [runs[k]["V"][-1] / 1e6 for k in keys]}
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.2))
    for a, (t, v) in zip(ax, vals.items()):
        a.bar(lab, v, color=col, edgecolor="k"); a.set_title(t, loc="left"); a.tick_params(axis="x", rotation=30)
        for i, x in enumerate(v): a.text(i, x, f"{x:.1f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Comparaison des stratégies de dispatch — consigne {P_label} (A : hydro seul, B : hybride)", fontweight="bold")
    return save_figure(fig, d / "fig_p3_01_comparaison_globale")


def fig_reservoir_dyn(runs, cas, idx, res, d, name, titre):
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for strat, c in (("RBC", BLEU), ("DP", VERT)):
        o = runs[(cas, strat)]
        ax[0].plot(idx, o["V"][:-1] / 1e6, color=c, label=strat); ax[1].plot(idx, _daily_arr(o["Q_tur"], idx, "mean").reindex(idx.normalize()).values, color=c, lw=0.8, label=strat)
    ax[0].axhline(res.V_min / 1e6, color="k", ls="--"); ax[0].set(ylabel="Volume (Mm³)"); ax[0].legend()
    ax[0].set_title(f"(a) {titre} — Volume", loc="left"); ax[1].set(ylabel="Q$_{tur}$ moyen journalier (m³/s)", xlabel="Date"); ax[1].set_title("(b) Débit turbiné", loc="left")
    fig.suptitle(f"Dynamique du réservoir — {titre}", fontweight="bold")
    return save_figure(fig, d / name)


def fig_deficit_series(runs, idx, d):
    fig, ax = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True, sharey=True)
    for a, (strat, c) in zip(ax, (("RBC", BLEU), ("DP", VERT))):
        a.plot(idx, runs[("hybride", strat)]["P_def"] / 1e6, color=c, lw=0.5); a.set_ylabel("P$_{def}$ (MW)"); a.set_title(f"({'a' if strat == 'RBC' else 'b'}) {strat} — Déficit horaire", loc="left")
    ax[-1].set_xlabel("Date"); fig.suptitle("Séries de déficit — scénario B (hybride)", fontweight="bold")
    return save_figure(fig, d / "fig_p3_04_deficit_timeseries_B")


def fig_boxplot_et_duree(runs, idx, d):
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [_daily_arr(runs[("hybride", s)]["P_def"] / 1e6, idx, "mean").values for s in ("RBC", "DP")]
    data = [x[x > 0] for x in data]
    ax.boxplot(data, tick_labels=["RBC", "DP"], showmeans=True); ax.set_ylabel("Déficit moyen journalier (MW), jours déficitaires")
    ax.set_title("Distribution des déficits — scénario B")
    p1 = save_figure(fig, d / "fig_p3_05_boxplot_deficits")
    fig, ax = plt.subplots(figsize=(8, 5))
    for s, c in (("RBC", BLEU), ("DP", VERT)):
        x = np.sort(runs[("hybride", s)]["P_def"] / 1e6)[::-1]; ax.plot(np.arange(1, len(x) + 1) / len(x) * 100, x, color=c, label=s)
    ax.set(xlabel="Part du temps dépassée (%)", ylabel="P$_{def}$ (MW)"); ax.legend(); ax.set_title("Courbe de durée du déficit — scénario B")
    save_figure(fig, d / "fig_p3_06_courbe_duree")
    return p1


def fig_surcharge(kpi_dp, d):
    """kpi_dp : DataFrame de kpi_dp_vs_rbc.csv ; déficit & volume final vs taux de surcharge (réf. 60 MW)."""
    k = kpi_dp.assign(taux=(kpi_dp["P_dem_MW"] / 60 - 1) * 100)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for cas, a, t in (("hydro_seul", ax[0], "(a) Déficit — hydro seul (A)"), ("hybride", ax[1], "(b) Déficit — hybride (B)")):
        for s, c, m in (("RBC", BLEU, "o"), ("DP", VERT, "s")):
            q = k[(k.cas == cas) & (k.strategie == s)]; a.plot(q["taux"], q["E_def_GWh"], "-" + m, color=c, label=s)
        a.set(xlabel="Taux de surcharge par rapport à 60 MW (%)", ylabel="Déficit cumulé (GWh)"); a.set_title(t, loc="left"); a.legend()
    q = k[(k.strategie == "RBC")]
    for cas, c, lab in (("hydro_seul", BLEU, "A hydro seul"), ("hybride", ORANGE, "B hybride")):
        z = q[q.cas == cas]; ax[2].plot(z["taux"], z["E_def_GWh"] / z["E_dem_GWh"] * 100, "-o", color=c, label=lab)
    ax[2].set(xlabel="Taux de surcharge (%)", ylabel="Déficit / énergie demandée (%)"); ax[2].set_title("(c) Part de demande non servie (RBC)", loc="left"); ax[2].legend()
    fig.suptitle("Comportement sous surcharge (consigne constante)", fontweight="bold")
    return save_figure(fig, d / "fig_p3_07_surcharge")


def fig_chap6_hydro(runs, idx, d):
    o = runs[("hydro_seul", "RBC")]
    fig, ax = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True)
    ax[0].plot(idx, pd.Series(o["P_H"] / 1e6, index=idx).rolling(24, min_periods=1).mean(), color=BLEU); ax[0].set_ylabel("P$_H$ (MW)")
    ax[0].set_title("(a) Puissance hydro (moyenne 24 h) — consigne 60 MW, hydro seul", loc="left")
    ax[1].plot(idx, pd.Series(o["Q_tur"], index=idx).rolling(24, min_periods=1).mean(), color=VERT); ax[1].set_ylabel("Q$_{tur}$ (m³/s)"); ax[1].set_title("(b) Débit turbiné", loc="left")
    ax[2].plot(idx, o["Q_spill"], color=ROUGE); ax[2].set(ylabel="Q$_{dév}$ (m³/s)", xlabel="Date"); ax[2].set_title("(c) Déversement", loc="left")
    fig.suptitle("6.2 — Centrale hydroélectrique : production et déversement", fontweight="bold")
    return save_figure(fig, d / "fig_ch6_01_hydro")


def fig_chap6_dispatch(runs, idx, d, jours=7, debut="2025-06-02"):
    sl = (idx >= pd.Timestamp(debut)) & (idx < pd.Timestamp(debut) + pd.Timedelta(days=jours))
    fig, ax = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    for a, cas, t in ((ax[0], "hydro_seul", "(a) Scénario A (60 MW, hydro seul)"), (ax[1], "hybride", "(b) Scénario B (60 MW + FPV)")):
        for s, c, m in (("RBC", BLEU, "-o"), ("DP", VERT, "-s")):
            a.plot(idx[sl], runs[(cas, s)]["P_H"][sl] / 1e6, m, color=c, ms=3, lw=1.2, label=s)
        if cas == "hybride": a.plot(idx[sl], runs[(cas, "RBC")]["P_fpv"][sl] / 1e6, color=ORANGE, lw=1.0, ls="--", label="FPV")
        a.set_ylabel("P$_H$ (MW)"); a.set_title(t, loc="left"); a.legend(ncol=3)
    ax[-1].set_xlabel("Heure")
    fig.suptitle(f"6.4 — Comparaison des décisions de dispatch ({jours} jours dès {debut})", fontweight="bold")
    return save_figure(fig, d / "fig_ch6_02_dispatch_comparaison")


# ---------------------------------------------------------------------------------------------
# 4 stratégies (RBC, DP, MILP, SMPC) et 12 scénarios hydro-climatiques (visualisation_part3/part4/surcharge)
# ---------------------------------------------------------------------------------------------
STRAT = ("RBC", "DP", "MILP", "SMPC"); COL = {"RBC": BLEU, "DP": VERT, "MILP": ORANGE, "SMPC": ROUGE}
MARK = {"RBC": "o", "DP": "s", "MILP": "^", "SMPC": "D"}


def fig_matrice_strategies(M, d):
    """Déficit annuel par consigne pour les 4 stratégies, hydro seul (A) et hybride (B), base."""
    q = M[(M.stress == "base") & (M.charge == "ferme")]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for a, cas, t in ((ax[0], "hydro_seul", "(a) Hydro seul (A)"), (ax[1], "hybride", "(b) Hybride FPV + hydro (B)")):
        for s in STRAT:
            z = q[(q.cas == cas) & (q.strategie == s)].sort_values("P_dem_MW")
            a.plot(z["P_dem_MW"], z["E_def_GWh"], "-" + MARK[s], color=COL[s], label=s)
        a.set(xlabel="Consigne ferme P$_{dem}$ (MW)"); a.set_title(t, loc="left"); a.legend(ncol=2)
    ax[0].set_ylabel("Déficit annuel (GWh)")
    fig.suptitle("Comparaison des 4 stratégies de dispatch — consigne constante, 2025 (11 mois)", fontweight="bold")
    return save_figure(fig, d / "fig_p3_08_strategies_vs_consigne")


def fig_strat_60_75(M, d):
    q = M[(M.stress == "base") & (M.charge == "ferme") & (M.P_dem_MW.isin([60.0, 75.0]))]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    cols = [("E_def_GWh", "Déficit (GWh)"), ("R_comp", "R$_{comp}$ (–)"), ("V_min_obs_Mm3", "V$_{min}$ atteint (Mm³)")]
    for i, P in enumerate((60.0, 75.0)):
        for j, (c, lab) in enumerate(cols):
            a = ax[i, j]; w = 0.38
            for k, cas in enumerate(("hydro_seul", "hybride")):
                v = [float(q[(q.P_dem_MW == P) & (q.cas == cas) & (q.strategie == s)][c].iloc[0]) for s in STRAT]
                a.bar(np.arange(4) + (k - 0.5) * w, v, w, label="A hydro seul" if k == 0 else "B hybride", color=[BLEU, ORANGE][k], edgecolor="k")
            a.set_xticks(range(4)); a.set_xticklabels(STRAT); a.set_ylabel(lab); a.set_title(f"{P:.0f} MW", loc="left")
            if i == 0 and j == 0: a.legend()
    fig.suptitle("Indicateurs par stratégie — consignes 60 et 75 MW", fontweight="bold")
    return save_figure(fig, d / "fig_p3_09_indicateurs_strategies")


def fig_scen_effectifs(sc, info, d):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for a, h, t in ((ax[0], "MaxHydric", f"Max Hydric (V > {info['V_median_Mm3']:.0f} Mm³)"), (ax[1], "MinHydric", f"Min Hydric (V ≤ {info['V_median_Mm3']:.0f} Mm³)")):
        m = np.array([[int(sc[(sc.saison == s) & (sc.hydro == h) & (sc.pv == p)].n_jours.iloc[0]) for p in ("Sunny", "Cloudy", "Rainy")] for s in ("Wet", "Dry")])
        im = a.imshow(m, cmap="viridis"); a.grid(False)
        a.set_xticks(range(3)); a.set_xticklabels(["Sunny", "Cloudy", "Rainy"]); a.set_yticks(range(2)); a.set_yticklabels(["Wet", "Dry"])
        for i in range(2):
            for j in range(3):
                a.text(j, i, str(m[i, j]) + ("" if m[i, j] >= 15 else "\n(<15 j)"), ha="center", va="center", color="w" if m[i, j] < m.max() * 0.6 else "k", fontsize=12, fontweight="bold")
        a.set_title(t, loc="left"); fig.colorbar(im, ax=a, shrink=0.8)
    fig.suptitle("Effectifs des 12 scénarios hydro-climatiques (jours, 2025)", fontweight="bold")
    return save_figure(fig, d / "fig_p4_01_effectifs_scenarios")


def _viables(S):
    return S[S.viable]


def fig_scen_deficit(S, d, P=75.0):
    q = _viables(S[(S.stress == "base") & (S.charge == "ferme") & (S.P_dem_MW == P)])
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.2), sharey=True)
    for a, cas, t in ((ax[0], "hydro_seul", "(a) Hydro seul"), (ax[1], "hybride", "(b) Hybride")):
        ids = sorted(q.scenario_id.unique()); w = 0.2
        for k, s in enumerate(STRAT):
            v = [float(q[(q.cas == cas) & (q.strategie == s) & (q.scenario_id == i)].E_def_GWh.iloc[0]) for i in ids]
            a.bar(np.arange(len(ids)) + (k - 1.5) * w, v, w, color=COL[s], label=s, edgecolor="k", lw=0.4)
        a.set_xticks(range(len(ids))); a.set_xticklabels([f"S{i}" for i in ids]); a.set_title(t, loc="left")
        a.set_xlabel("Scénario (viables, ≥ 15 j)")
        if cas == "hydro_seul": a.legend(ncol=2)
    ax[0].set_ylabel("Déficit sur les jours du scénario (GWh)")
    fig.suptitle(f"Déficit par scénario hydro-climatique — consigne {P:.0f} MW", fontweight="bold")
    return save_figure(fig, d / "fig_p4_02_deficit_scenarios")


def fig_scen_stress(S, d, P=75.0, strat="RBC"):
    q = _viables(S[(S.charge == "ferme") & (S.P_dem_MW == P) & (S.strategie == strat) & (S.cas == "hybride")])
    ids = sorted(q.scenario_id.unique()); w = 0.2
    fig, ax = plt.subplots(figsize=(13, 5))
    for k, (st, c) in enumerate((("base", BLEU), ("S1", VERT), ("S2", ORANGE), ("S3", ROUGE))):
        v = [float(q[(q.stress == st) & (q.scenario_id == i)].E_def_GWh.iloc[0]) for i in ids]
        ax.bar(np.arange(len(ids)) + (k - 1.5) * w, v, w, color=c, label={"base": "Base", "S1": "S1 panne 1 groupe", "S2": "S2 panne 2 groupes", "S3": "S3 sécheresse ×0,5"}[st], edgecolor="k", lw=0.4)
    ax.set_xticks(range(len(ids))); ax.set_xticklabels([f"S{i}" for i in ids]); ax.legend(ncol=2); ax.set_ylabel("Déficit (GWh)")
    ax.set_title(f"Déficit par scénario et par stress — hybride, {strat}, {P:.0f} MW")
    return save_figure(fig, d / "fig_p4_03_stress_comparaison")


def fig_scen_classement(S, d, P=75.0, stress="S2", strat="RBC"):
    q = _viables(S[(S.charge == "ferme") & (S.P_dem_MW == P) & (S.strategie == strat) & (S.cas == "hybride") & (S.stress == stress)]).copy()
    q["def_par_jour"] = q["E_def_GWh"] / q["n_jours"] * 1000     # MWh/jour
    q = q.sort_values("def_par_jour")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.barh([f"S{i} {n.split(': ')[1]}" for i, n in zip(q.scenario_id, q.scenario)], q["def_par_jour"], color=ROUGE, edgecolor="k")
    ax.set_xlabel("Déficit moyen par jour du scénario (MWh/j)")
    ax.set_title(f"Classement des scénarios par vulnérabilité ({stress}, hybride, {strat}, {P:.0f} MW)")
    ax.invert_yaxis()
    return save_figure(fig, d / "fig_p4_04_classement_vulnerabilite")


def fig_scen_robustesse(S, d, strat="RBC", cas="hybride", ids=None):
    q = _viables(S[(S.charge == "ferme") & (S.stress == "base") & (S.strategie == strat) & (S.cas == cas)])
    sel = ids or [2, 6, 8, 10]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in sel:
        z = q[q.scenario_id == i].sort_values("P_dem_MW")
        if len(z): ax.plot(z["P_dem_MW"], z["E_def_GWh"] / z["E_dem_GWh"] * 100, "-o", label=z["scenario"].iloc[0].split(": ")[1])
    ax.set(xlabel="Consigne (MW)", ylabel="Part de demande non servie (%)"); ax.legend(fontsize=9)
    ax.set_title(f"Déficit vs consigne pour des scénarios représentatifs ({cas}, {strat})")
    return save_figure(fig, d / "fig_p4_05_robustesse_vs_pdem")


def fig_surcharge4(M, d):
    k = M[(M.stress == "base") & (M.charge == "ferme")]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for cas, a, t in (("hydro_seul", ax[0], "(a) Déficit — hydro seul (A)"), ("hybride", ax[1], "(b) Déficit — hybride (B)")):
        for s in STRAT:
            z = k[(k.cas == cas) & (k.strategie == s)].sort_values("P_dem_MW")
            a.plot((z["P_dem_MW"] / 60 - 1) * 100, z["E_def_GWh"], "-" + MARK[s], color=COL[s], label=s)
        a.set(xlabel="Taux de surcharge par rapport à 60 MW (%)", ylabel="Déficit cumulé (GWh)"); a.set_title(t, loc="left"); a.legend(ncol=2)
    for cas, c, lab in (("hydro_seul", BLEU, "A hydro seul"), ("hybride", ORANGE, "B hybride")):
        z = k[(k.cas == cas) & (k.strategie == "DP")].sort_values("P_dem_MW")
        ax[2].plot((z["P_dem_MW"] / 60 - 1) * 100, z["E_def_GWh"] / z["E_dem_GWh"] * 100, "-o", color=c, label=lab)
    ax[2].set(xlabel="Taux de surcharge (%)", ylabel="Déficit / énergie demandée (%)"); ax[2].set_title("(c) Part non servie (DP)", loc="left"); ax[2].legend()
    fig.suptitle("Comportement sous surcharge, jusqu'à +50 % de la puissance ferme (consigne constante)", fontweight="bold")
    return save_figure(fig, d / "fig_p3_07_surcharge")


def fig_daily_strategies(J, d, P=75.0):
    q = J[J.P_dem_MW == P]
    fig, ax = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
    for j, cas in enumerate(("hydro_seul", "hybride")):
        for s in STRAT:
            z = q[(q.cas == cas) & (q.strategie == s)]
            t = pd.to_datetime(z["date"])
            ax[0, j].plot(t, z["V_fin_Mm3"], color=COL[s], label=s, lw=1.4)
            ax[1, j].plot(t, z["P_def_MW"].rolling(7, min_periods=1).mean(), color=COL[s], lw=1.1, label=s)
        ax[0, j].set_title(("(a) Hydro seul" if j == 0 else "(b) Hybride") + " — volume", loc="left"); ax[0, j].set_ylabel("V (Mm³)")
        ax[1, j].set_title("Déficit moyen 7 j", loc="left"); ax[1, j].set_ylabel("P$_{def}$ (MW)")
    ax[0, 0].legend(ncol=2)
    fig.suptitle(f"Dynamique du réservoir et déficit — 4 stratégies, consigne {P:.0f} MW", fontweight="bold")
    return save_figure(fig, d / "fig_p3_10_dynamique_4strategies")


def fig_couverture(Cv, d):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for P, ls in ((60.0, "-"), (75.0, "--")):
        for s, c in (("RBC", BLEU), ("DP", VERT)):
            z = Cv[(Cv.P_dem_MW == P) & (Cv.strategie == s)].sort_values("f_cov_pct")
            ax[0].plot(z["f_cov_pct"], z["E_def_GWh"], ls, marker="o", color=c, label=f"{s} {P:.0f} MW")
    ax[0].set(xlabel="Taux de couverture f$_{cov}$ de la surface à V$_{min}$ (%)", ylabel="Déficit annuel (GWh)", xscale="symlog", xlim=(0, 35))
    ax[0].axvline(1.43, color="0.4", ls=":"); ax[0].text(1.5, ax[0].get_ylim()[1] * 0.9, "dimension actuelle\n(96,75 MWp, 1,4 %)", fontsize=8)
    ax[0].legend(fontsize=8); ax[0].set_title("(a) Déficit vs couverture", loc="left")
    z = Cv[(Cv.P_dem_MW == 75) & (Cv.strategie == "RBC")].sort_values("f_cov_pct")
    ax[1].plot(z["f_cov_pct"], z["E_fpv_GWh"], "-o", color=ORANGE, label="Énergie FPV"); ax[1].plot(z["f_cov_pct"], z["E_curt_GWh"], "-s", color=ROUGE, label="Écrêtée (75 MW)")
    ax[1].set(xlabel="f$_{cov}$ (%)", ylabel="GWh", xscale="symlog", yscale="symlog", xlim=(0, 35)); ax[1].legend(); ax[1].set_title("(b) Énergie FPV et écrêtage", loc="left")
    ax[2].plot(z["f_cov_pct"], z["evap_evitee_Mm3"], "-o", color=CYAN); ax[2].set(xlabel="f$_{cov}$ (%)", ylabel="Évaporation évitée (Mm³ sur 11 mois)", xscale="symlog", xlim=(0, 35))
    ax[2].set_title("(c) Eau d'évaporation économisée", loc="left")
    fig.suptitle("Sensibilité au taux de couverture FPV", fontweight="bold")
    return save_figure(fig, d / "fig_p5_01_sensibilite_couverture")


def fig_sobol(So, d):
    fig, ax = plt.subplots(figsize=(8, 4.6)); x = np.arange(len(So)); w = 0.38
    ax.bar(x - w / 2, So["S1"], w, color=BLEU, label="S1 (premier ordre)", edgecolor="k"); ax.bar(x + w / 2, So["ST"], w, color=ORANGE, label="ST (total)", edgecolor="k")
    ax.set_xticks(x); ax.set_xticklabels(So["facteur"]); ax.set_ylabel("Indice de Sobol"); ax.legend()
    ax.set_title("Sobol — déficit annuel (hybride, 75 MW, RBC) ; N = 64, indicatif")
    return save_figure(fig, d / "fig_p5_02_sobol")


def fig_potentiel(D, d, E_hydro_GWh=332.0):
    f = D["f_cov_pct"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0, 0].plot(f, D["P_DC_MWp"], "-o", color=BLEU); ax[0, 0].set(xlabel="Couverture de S$_{min}$ (%)", ylabel="Puissance DC installable (MWp)")
    ax[0, 0].set_title("(a) Puissance installable (S$_{min}$ = 35,1 km²)", loc="left")
    for x, y, a in zip(f, D["P_DC_MWp"], D["A_km2"]):
        if x in (1.0, 5.0, 10.0, 20.0, 30.0): ax[0, 0].annotate(f"{a:.1f} km²", (x, y), textcoords="offset points", xytext=(-8, 8), ha="right", fontsize=8, color="0.35")
    ax[0, 1].plot(f, D["E_an_GWh_A_proportionnel"], "-o", color=ORANGE, label="(A) onduleurs proportionnels"); ax[0, 1].plot(f, D["E_an_GWh_B_fixe75"], "-s", color=ROUGE, label="(B) évacuation fixe 75 MW")
    ax[0, 1].axhline(E_hydro_GWh, color=VERT, ls="--"); ax[0, 1].text(f.iloc[0], E_hydro_GWh * 1.05, "hydro simulée (≈ 332 GWh / 11 mois)", color=VERT, fontsize=9)
    ax[0, 1].set(xlabel="Couverture (%)", ylabel="Énergie FPV 2025 (GWh/an)", yscale="log"); ax[0, 1].legend(fontsize=9); ax[0, 1].set_title("(b) Énergie annuelle", loc="left")
    for P, ls in ((60, "-"), (75, "--")):
        ax[1, 0].plot(f, D[f"def_GWh_{P}MW_DP_B_fixe75"], ls, marker="o", color=VERT, label=f"DP {P} MW"); ax[1, 0].plot(f, D[f"def_GWh_{P}MW_RBC_B_fixe75"], ls, marker="s", color=BLEU, label=f"RBC {P} MW")
    ax[1, 0].set(xlabel="Couverture (%)", ylabel="Déficit (GWh, 11 mois)"); ax[1, 0].legend(fontsize=9, ncol=2); ax[1, 0].set_title("(c) Déficit résiduel (évacuation 75 MW)", loc="left")
    ax[1, 1].plot(f, D["evap_evitee_Mm3_an"], "-o", color=CYAN); ax[1, 1].set(xlabel="Couverture (%)", ylabel="Évaporation évitée (Mm³/an)")
    ax[1, 1].set_title("(d) Eau économisée (négligeable devant ≈ 1 180 Mm³ turbinés)", loc="left")
    fig.suptitle("Potentiel FPV sur la retenue en fonction du taux de couverture", fontweight="bold")
    return save_figure(fig, d / "fig_p5_03_potentiel_fpv")
