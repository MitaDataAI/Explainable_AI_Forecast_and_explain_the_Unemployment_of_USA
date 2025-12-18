import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.dates as mdates


from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm


# ======================================================
# Constantes ET Helpers
# ======================================================
ALPHA = 0.05

def wide_to_long_timeseries(
    df: pd.DataFrame,
    *,
    type_label: str = "Original",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    DataFrame wide (index=date, colonnes=séries) -> format long.
    Output: colonnes [date, series, value, type]
    """
    return (
        df.copy()
          .rename_axis(date_col)
          .reset_index()
          .assign(**{date_col: lambda x: pd.to_datetime(x[date_col])})
          .melt(id_vars=date_col, var_name="series", value_name="value")
          .dropna(subset=["value"])
          .sort_values(["series", date_col])
          .assign(type=type_label)
          .reset_index(drop=True)
    )

# ======================================================
# 1) Visualisation – séries originales
# ======================================================


def plot_original_series(
    df: pd.DataFrame,
    col_wrap: int = 4,
    linewidth: float = 1.0,
    style: str = "dark_background",
):
    """
    Trace les séries temporelles originales
    (conversion wide → long via wide_to_long_timeseries).
    """
    df_long_orig = wide_to_long_timeseries(df, type_label="Original")

    plt.style.use(style)

    g = sns.relplot(
        data=df_long_orig,
        x="date",
        y="value",
        col="series",
        col_wrap=col_wrap,
        kind="line",
        linewidth=linewidth,
        facet_kws=dict(sharey=False),
        legend=False,
    )

    g.set_axis_labels("", "")
    g.set_titles("{col_name}")

    for ax in g.axes.flat:
        ax.tick_params(axis="x", labelrotation=0)
        ax.grid(False)

    plt.show()
    return df_long_orig


# ======================================================
# 2) Helpers de transformation (diff / logdiff)
# ======================================================
def diff_l(series, l: int) -> pd.Series:
    """
    Différence au lag l : y_t - y_{t-l}
    (NB: ce n'est pas une "diff répétée l fois", mais un pas de l périodes)
    """
    s = pd.Series(series).astype(float)
    return s - s.shift(int(l))


def log_diff(series: pd.Series, lags: int = 1) -> pd.Series:
    """
    Log-diff: log(y_t) - log(y_{t-lags}).
    Les valeurs <= 0 sont mises à NaN (log non défini).
    """
    s = pd.Series(series).astype(float)
    s = s.where(s > 0)  # <=0 -> NaN
    return np.log(s).diff(int(lags))


# ======================================================
# 3) ADF standardisé (UNIQUE wrapper)
# ======================================================
def adf_test(
    series: pd.Series,
    *,
    alpha: float = ALPHA,
    regression: str = "ct",
    autolag: str = "AIC",
    min_n: int = 20,
) -> dict:
    """
    Test ADF standardisé.
    Gère le cas série constante (ADF impossible) et évite le crash.
    """
    y = (
        pd.Series(series)
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    nobs = int(len(y))
    if nobs < min_n:
        return {"ok": False, "reason": "insufficient_data", "nobs": nobs}

    # ✅ Série constante -> ADF impossible
    if y.nunique(dropna=True) <= 1:
        return {"ok": False, "reason": "constant_series", "nobs": nobs}

    try:
        stat, pval, usedlag, nobs2, crit, icbest = adfuller(
            y, regression=regression, autolag=autolag
        )
    except ValueError as e:
        # ✅ Sécurité supplémentaire
        if "x is constant" in str(e).lower():
            return {"ok": False, "reason": "constant_series", "nobs": nobs}
        return {"ok": False, "reason": f"adf_error: {e}", "nobs": nobs}

    return {
        "ok": True,
        "stat": float(stat),
        "pval": float(pval),
        "lags": int(usedlag),
        "nobs": int(nobs2),
        "crit": crit,
        "stationary": bool(pval < alpha),
    }


def test_adf_trend(series, autolag: str = "AIC") -> dict:
    """
    Compat: ADF(tau3) avec constante + tendance (regression='ct').
    Retourne : {'adf_stat', 'adf_pval', 'usedlag'}.
    """
    res = adf_test(series, alpha=ALPHA, regression="ct", autolag=autolag)
    if not res.get("ok", False):
        return {"adf_stat": np.nan, "adf_pval": np.nan, "usedlag": None}
    return {"adf_stat": res["stat"], "adf_pval": res["pval"], "usedlag": res["lags"]}


# ======================================================
# 4) Choix d'un lag de diff via ADF(ct) sur Δ_l y_t
# ======================================================
def find_opt_diff_lag(
    series,
    candidates=(1, 3, 6, 12),
    *,
    alpha: float = ALPHA,
    autolag: str = "AIC",
    regression: str = "ct",
):
    """
    Recherche un lag l (parmi candidates) tel que ADF(ct) sur Δ_l y_t rejette H0.
    Renvoie (opt_lag, details, best_lag).

    - opt_lag : plus petit l qui passe (p<alpha), sinon None
    - best_lag: l qui minimise p-value (si dispo)
    """
    details = {}
    best_l, best_p = None, np.inf

    for l in candidates:
        dy = diff_l(series, l)
        res = adf_test(dy, alpha=alpha, regression=regression, autolag=autolag)
        pval = res.get("pval", np.nan) if res.get("ok", False) else np.nan
        usedlag = res.get("lags", None) if res.get("ok", False) else None

        details[int(l)] = {"pval": pval, "usedlag": usedlag}

        if pval is not None and not np.isnan(pval) and pval < best_p:
            best_p, best_l = pval, int(l)

    passing = [
        l for l, d in details.items()
        if d["pval"] is not None and not np.isnan(d["pval"]) and d["pval"] < alpha
    ]
    opt_lag = min(passing) if passing else None
    return opt_lag, details, best_l


# ======================================================
# 5) Modèle 1 – estimation OLS (ADF avec retards)
# ======================================================
def estimate_model1(series, lags: int):
    """
    Modèle 1 (ADF) :
    Δy_t = const + trend + y_{t-1} + Σ Δy_{t-i} + ε_t
    """
    y = pd.Series(series).astype(float).replace([np.inf, -np.inf], np.nan).dropna()

    dy = y.diff()
    y_lag = y.shift(1)
    trend = np.arange(1, len(y) + 1)

    X = pd.DataFrame({"const": 1.0, "trend": trend, "y_lag": y_lag}, index=y.index)

    for i in range(1, int(lags) + 1):
        X[f"dy_lag{i}"] = dy.shift(i)

    df_model = pd.concat([dy.rename("dy"), X], axis=1).dropna()
    return sm.OLS(df_model["dy"], df_model.drop(columns=["dy"])).fit()


def test_fisher_phi2(ols_res) -> dict:
    """
    Test Fisher φ2 (conjoint) :
    H0^2 : const = 0, trend = 0, y_lag = 0
    """
    f_test = ols_res.f_test("const = 0, trend = 0, y_lag = 0")
    return {"phi2_F": float(f_test.fvalue), "phi2_pval": float(f_test.pvalue)}


# ======================================================
# 6) Décision workflow : Nature + Conclusion (transformation)
# ======================================================
def _decide_nature_and_conclusion(
    adf_pval: float,
    beta1_pval: float,
    phi2_pval: float,
    *,
    alpha: float = ALPHA,
):
    """
    Nature (économétrique) :
      - Stationary (trend-stationary, trend significant)  [CAS B + beta1 sig]
      - Stationary (no significant trend)                 [CAS B + beta1 non sig]
      - Integrated I(1)                                   [CAS A + phi2 rejette]
      - Random Walk (RW)                                  [CAS A + phi2 non rejette]
      - TOO_SHORT

    Conclusion (action / transformation) :
      - detrend (remove deterministic trend)
      - none (already stationary)
      - diff(1)
      - model2 (DF model 2)   # signal: workflow indique passer au Modèle 2
      - unknown
    """
    if adf_pval is None or np.isnan(adf_pval):
        return "TOO_SHORT", "unknown"

    # CAS B : rejet H0 (pas de racine unitaire) -> stationnaire en niveau
    if adf_pval < alpha:
        if beta1_pval is not None and not np.isnan(beta1_pval) and beta1_pval < alpha:
            return "Stationary (trend-stationary, trend significant)", "detrend (remove deterministic trend)"
        return "Stationary (no significant trend)", "none (already stationary)"

    # CAS A : non-rejet H0 -> racine unitaire
    if phi2_pval is not None and not np.isnan(phi2_pval) and phi2_pval < alpha:
        return "Integrated I(1)", "diff(1)"

    return "Random Walk (RW)", "model2 (DF model 2)"


# ======================================================
# 7) Analyse stationnarité – modèle 1 (riche / essentiel)
# ======================================================
def analyse_series(series, alpha: float = ALPHA):
    """
    Analyse Modèle 1 (workflow DF):
    - ADF(ct) tau3: H0 gamma=0 (racine unitaire)
    - OLS model1 + test Student sur beta1 (trend)
    - Test Fisher φ2: H0^2 const=trend=y_lag=0
    - opt_diff_lag: diagnostic sur Δ_l y_t (ADF ct)
    - Renvoie Nature + Conclusion (transformation)
    """
    adf_out = test_adf_trend(series)

    if adf_out["usedlag"] is None:
        return {
            "ADF stat (tau3)": np.nan,
            "ADF p-value": np.nan,
            "lags used (ADF)": None,
            "beta1": np.nan,
            "beta1 p-value": np.nan,
            "phi2_F": np.nan,
            "phi2_pval": np.nan,
            "opt_diff_lag": None,
            "best_diff_lag": None,
            "Nature": "TOO_SHORT",
            "Conclusion": "unknown",
        }

    ols_res = estimate_model1(series, adf_out["usedlag"])
    beta1 = float(ols_res.params.get("trend", np.nan))
    beta1_p = float(ols_res.pvalues.get("trend", np.nan))

    fisher = test_fisher_phi2(ols_res)
    phi2_F = fisher["phi2_F"]
    phi2_p = fisher["phi2_pval"]

    opt_lag, _, best_lag = find_opt_diff_lag(series, alpha=alpha)

    nature, conclusion = _decide_nature_and_conclusion(
        adf_pval=adf_out["adf_pval"],
        beta1_pval=beta1_p,
        phi2_pval=phi2_p,
        alpha=alpha,
    )

    return {
        "ADF stat (tau3)": adf_out["adf_stat"],
        "ADF p-value": adf_out["adf_pval"],
        "lags used (ADF)": adf_out["usedlag"],
        "beta1": beta1,
        "beta1 p-value": beta1_p,
        "phi2_F": phi2_F,
        "phi2_pval": phi2_p,
        "opt_diff_lag": opt_lag,
        "best_diff_lag": best_lag,
        "Nature": nature,
        "Conclusion": conclusion,
    }


def analyse_series_essentiel(series, alpha: float = ALPHA):
    """
    Version synthèse : garde l'essentiel + Nature + Conclusion.
    """
    adf_out = test_adf_trend(series)

    if adf_out["usedlag"] is None:
        return {
            "ADF p-value": np.nan,
            "lags used (ADF)": None,
            "beta1 p-value": np.nan,
            "phi2_pval": np.nan,
            "opt_diff_lag": None,
            "Nature": "TOO_SHORT",
            "Conclusion": "unknown",
        }

    ols_res = estimate_model1(series, adf_out["usedlag"])
    beta1_p = float(ols_res.pvalues.get("trend", np.nan))

    fisher = test_fisher_phi2(ols_res)
    phi2_p = fisher["phi2_pval"]

    opt_lag, _, _ = find_opt_diff_lag(series, alpha=alpha)

    nature, conclusion = _decide_nature_and_conclusion(
        adf_pval=adf_out["adf_pval"],
        beta1_pval=beta1_p,
        phi2_pval=phi2_p,
        alpha=alpha,
    )

    return {
        "ADF p-value": adf_out["adf_pval"],
        "lags used (ADF)": adf_out["usedlag"],
        "beta1 p-value": beta1_p,
        "phi2_pval": phi2_p,
        "opt_diff_lag": opt_lag,
        "Nature": nature,
        "Conclusion": conclusion,
    }


def analyse_stationnarite(df: pd.DataFrame, drop_cols=("USREC",), alpha: float = ALPHA):
    """
    Analyse riche sur toutes les séries d'un DataFrame wide.
    Retour: DataFrame indexé par le nom de série (pas de colonne 'series').
    """
    out = {}
    for col in df.columns:
        if col in drop_cols:
            continue
        out[col] = analyse_series(df[col], alpha=alpha)
    return pd.DataFrame(out).T


def analyse_stationnarite_essentiel(df: pd.DataFrame, drop_cols=("USREC",), alpha: float = ALPHA):
    """
    Analyse essentielle sur toutes les séries d'un DataFrame wide.
    Retour: DataFrame indexé par le nom de série (pas de colonne 'series').
    """
    out = {}
    for col in df.columns:
        if col in drop_cols:
            continue
        out[col] = analyse_series_essentiel(df[col], alpha=alpha)
    return pd.DataFrame(out).T


# ======================================================
# 8) Stationnarisation rules-based + rapport ADF (candidats)
# ======================================================
DEFAULT_STATIONARIZE_RULES = {
    "_default": {"method": "logdiff", "lags": 1},
}


def choose_rule(col_name: str, rules: dict | None) -> dict:
    """
    Sélectionne la règle associée à une colonne, sinon la règle _default.
    """
    if rules is None:
        rules = DEFAULT_STATIONARIZE_RULES
    if "_default" not in rules:
        raise ValueError("rules doit contenir une clé '_default'.")
    return rules.get(col_name, rules["_default"])


def apply_transform_one(
    col_name: str,
    series: pd.Series,
    rules: dict | None,
) -> tuple[pd.Series, str, int, str]:
    """
    Applique la règle de transformation d'une série.
    Retourne (serie_transformée, label_lisible, lags, method).
    """
    rule = choose_rule(col_name, rules)
    method, lags = rule["method"], int(rule["lags"])

    if method == "diff":
        y = diff_l(series, lags)
        label = f"diff(lag={lags})"

    elif method == "logdiff":
        y = log_diff(series, lags)
        label = f"logdiff(lag={lags})"

    elif method == "none":
        y = pd.Series(series).astype(float)
        label = "none"
        lags = 0

    else:
        raise ValueError(f"Méthode inconnue: {method} (col={col_name})")

    return y, label, lags, method


def adf_summary_one(
    col_name: str,
    series_trans: pd.Series,
    *,
    alpha: float,
    regression: str,
    autolag: str = "AIC",
) -> dict:
    """
    Lance un test ADF sur une série transformée et renvoie une ligne de résumé.
    Robuste au cas "x is constant" (ADF impossible).
    """
    try:
        res = adf_test(series_trans, alpha=alpha, regression=regression, autolag=autolag)
    except ValueError as e:
        if "x is constant" in str(e).lower():
            y = (
                pd.Series(series_trans)
                .astype(float)
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            return {
                "series": col_name,
                "ADF stat": np.nan,
                "p-value": np.nan,
                "usedlag(ADF)": np.nan,
                "nobs": int(len(y)),
                "verdict": "constant_series",
            }
        raise

    if res.get("ok", False):
        verdict = "Stationary" if res["stationary"] else "Non-stationary"
        return {
            "series": col_name,
            "ADF stat": res["stat"],
            "p-value": res["pval"],
            "usedlag(ADF)": res["lags"],
            "nobs": res["nobs"],
            "verdict": verdict,
        }

    return {
        "series": col_name,
        "ADF stat": np.nan,
        "p-value": np.nan,
        "usedlag(ADF)": np.nan,
        "nobs": res.get("nobs", np.nan),
        "verdict": res.get("reason", "error"),
    }


def stationarize_by_rules(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    rules: dict | None = None,
    alpha: float = ALPHA,
    adf_regression: str = "ct",
    autolag: str = "AIC",
    dropna_all: bool = True,
    skip_cols: tuple[str, ...] = ("USREC",),   # ✅ NEW: variables à ne pas stationnariser
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Applique les transformations (diff/logdiff/none), lance ADF et renvoie :
    (df_trans, summary_adf, meta)

    skip_cols:
      - colonnes conservées telles quelles (none), ADF non exécuté, verdict='skipped'
    """
    if columns is None:
        columns = list(df.columns)

    skip_set = set(skip_cols or ())

    df_trans = pd.DataFrame(index=df.index)
    meta: dict = {}
    rows: list[dict] = []

    for col in columns:
        if col not in df.columns:
            continue

        # ✅ Exception: on ne stationnarise pas (dummy / exogènes / etc.)
        if col in skip_set:
            y = pd.Series(df[col]).astype(float)
            df_trans[col] = y
            meta[col] = {"label": "none (skipped)", "lags": 0, "method": "skip"}

            y_clean = y.replace([np.inf, -np.inf], np.nan).dropna()
            rows.append({
                "series": col,
                "ADF stat": np.nan,
                "p-value": np.nan,
                "usedlag(ADF)": np.nan,
                "nobs": int(len(y_clean)),
                "verdict": "skipped",
            })
            continue

        # comportement normal
        y, label, lags, method = apply_transform_one(col, df[col], rules)
        df_trans[col] = y
        meta[col] = {"label": label, "lags": lags, "method": method}

        rows.append(
            adf_summary_one(
                col,
                y,
                alpha=alpha,
                regression=adf_regression,
                autolag=autolag,
            )
        )

    if dropna_all:
        df_trans = df_trans.replace([np.inf, -np.inf], np.nan).dropna(how="any")

    summary_adf = pd.DataFrame(rows).set_index("series").sort_index()
    return df_trans, summary_adf, meta


def build_stationarity_report(summary_adf: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """
    Construit un DataFrame unique et lisible à partir de summary_adf + meta.
    """
    results_df = summary_adf.copy()
    results_df["method"] = [meta[s]["label"] for s in results_df.index]

    results_df["status"] = np.select(
        [
            results_df["verdict"].eq("Stationary"),
            results_df["verdict"].eq("skipped"),
        ],
        [
            "✅ Stationary",
            "⏭️ Skipped (not tested)",
        ],
        default="❌ Non-stationary",
    )

    return results_df[["method", "ADF stat", "p-value", "usedlag(ADF)", "nobs", "status"]]


# ======================================================
# 9) ACF / PACF – séries stationnaires
# ======================================================
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_acf_pacf_stationary(
    df_stationary: pd.DataFrame,
    *,
    lags: int = 40,
    min_n: int = 30,
    figsize_per_series: tuple[float, float] = (12, 3),
    zero: bool = False,
):
    """
    Trace ACF (colonne gauche) et PACF (colonne droite) pour chaque série d'un
    DataFrame stationnaire.

    - Ignore les séries trop courtes (len < min_n) en affichant "too short".
    - Protège contre inf / -inf et NaN.
    - Ajuste automatiquement lags à min(lags, T//2) pour éviter les erreurs.

    Paramètres
    ----------
    df_stationary : pd.DataFrame
        DataFrame contenant les séries stationnarisées (colonnes).
    lags : int
        Nombre max de retards pour ACF / PACF (sera borné à T//2).
    min_n : int
        Nombre minimum d'observations non manquantes pour tracer.
    figsize_per_series : (w, h)
        Taille de figure par série (largeur, hauteur). La figure totale aura h*n_series.
    zero : bool
        Si True, inclut le lag 0 dans ACF/PACF. Par défaut False (comme ton snippet).
    """
    cols = list(df_stationary.columns)
    n_series = len(cols)

    if n_series == 0:
        raise ValueError("df_stationary ne contient aucune colonne à tracer.")

    fig, axes = plt.subplots(
        n_series, 2,
        figsize=(figsize_per_series[0], figsize_per_series[1] * n_series),
        squeeze=False,
    )

    for i, col in enumerate(cols):
        y = (
            pd.Series(df_stationary[col])
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if len(y) < min_n:
            axes[i, 0].set_title(f"ACF - {col} (too short: n={len(y)})")
            axes[i, 1].set_title(f"PACF - {col} (too short: n={len(y)})")
            axes[i, 0].axis("off")
            axes[i, 1].axis("off")
            continue

        max_lags = min(int(lags), max(1, len(y) // 2))

        # ACF à gauche
        plot_acf(y, lags=max_lags, ax=axes[i, 0], zero=zero)
        axes[i, 0].set_title(f"ACF - {col}")

        # PACF à droite
        plot_pacf(y, lags=max_lags, ax=axes[i, 1], zero=zero, method="ywm")
        axes[i, 1].set_title(f"PACF - {col}")

    plt.tight_layout()
    plt.show()

# ======================================================
# 10) Volatilité glissante (rolling std) – sélection flexible
# ======================================================

def plot_rolling_volatility(
    df_stationary: pd.DataFrame,
    *,
    window: int = 12,
    include_cols: list[str] | tuple[str, ...] | None = None,
    exclude_cols: list[str] | tuple[str, ...] | None = None,
    shock_bands: list[tuple[str, str]] | None = None,
    recession_col: str | None = "USREC",
    recession_source: pd.DataFrame | None = None,   # ✅ NEW (si USREC doit venir d'ailleurs)
    recession_alpha: float = 0.25,
    recession_color: str = "grey",
    add_recession_legend: bool = True,              # ✅ NEW
    to_month_start: bool = True,
    figsize: tuple[float, float] = (12, 5),
    linewidth: float = 1.0,
    style: str = "dark_background",
    title: str | None = None,
    return_data: bool = False,
):
    cols_all = list(df_stationary.columns)

    # 1) Sélection des colonnes
    if include_cols is not None:
        cols = [c for c in include_cols if c in cols_all]
    else:
        cols = cols_all.copy()

    if exclude_cols is not None:
        excl = set(exclude_cols)
        cols = [c for c in cols if c not in excl]

    # On ne trace généralement pas USREC comme série de volatilité
    if recession_col is not None and recession_col in cols:
        cols = [c for c in cols if c != recession_col]

    if len(cols) == 0:
        raise ValueError("Aucune colonne à tracer après sélection.")

    # 2) Rolling std
    df_sel = df_stationary[cols]
    roll_vol = df_sel.rolling(window).std()

    # 3) Index temporel propre
    roll_vol.index = pd.to_datetime(roll_vol.index)
    if to_month_start:
        roll_vol.index = roll_vol.index.to_period("M").to_timestamp()

    # 4) Plot
    plt.style.use(style)
    ax = roll_vol.plot(
        figsize=figsize,
        linewidth=linewidth,
        title=title or f"Volatilité glissante ({window} mois)",
    )

    # 5) Bandes de récession depuis USREC
    rec_df = recession_source if recession_source is not None else df_stationary
    recession_drawn = False

    if recession_col is not None and recession_col in rec_df.columns:
        usrec = rec_df[recession_col].copy()
        usrec.index = pd.to_datetime(usrec.index)
        if to_month_start:
            usrec.index = usrec.index.to_period("M").to_timestamp()

        # Aligner sur l’axe du plot (mêmes dates que roll_vol)
        usrec = usrec.reindex(roll_vol.index).astype(float).fillna(0.0)

        in_rec = (usrec.values == 1.0)
        idx = usrec.index

        start = None
        for t, flag in zip(idx, in_rec):
            if flag and start is None:
                start = t
            elif (not flag) and start is not None:
                ax.axvspan(start, t, color=recession_color, alpha=recession_alpha, zorder=0)
                recession_drawn = True
                start = None

        # si la série finit en récession
        if start is not None:
            ax.axvspan(start, idx[-1], color=recession_color, alpha=recession_alpha, zorder=0)
            recession_drawn = True

    # 6) Bandes manuelles (option)
    if shock_bands is not None:
        for start, end in shock_bands:
            ax.axvspan(
                pd.to_datetime(start),
                pd.to_datetime(end),
                color="grey",
                alpha=0.3,
                zorder=0,
            )

    # 7) Labels / grille
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Rolling std ({window} mois)")
    ax.grid(False)

    # 8) Légende avec récession
    if add_recession_legend and recession_drawn:
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor=recession_color, alpha=recession_alpha, label="Récession (USREC)"))
        ax.legend(handles=handles, labels=[*labels, "Récession (USREC)"], loc="upper left")

    plt.tight_layout()
    plt.show()

    if return_data:
        return roll_vol
    return None

# ======================================================
# 11) Original vs Stationnaire – côte à côte
#     (réutilise wide_to_long_timeseries + helper de style de plot_original_series)
# ======================================================

def plot_original_vs_stationary_side_by_side(
    df_original: pd.DataFrame,
    df_stationary: pd.DataFrame,
    *,
    meta: dict | None = None,  # meta renvoyé par stationarize_by_rules
    linewidth: float = 1.0,
    style: str = "dark_background",
    sharey: bool = False,
    height: float = 1.6,
    aspect: float = 3.0,
    skip_cols: list[str] | tuple[str, ...] | None = ("USREC",),
):
    """
    Trace chaque série sur une ligne : Original (gauche) vs Stationnaire (droite).
    - Conversion wide -> long via wide_to_long_timeseries (pas de redondance).
    - Style cohérent avec plot_original_series.
    - Titres par ligne basés sur meta (optionnel).
    """

    # --- 0) Filtre colonnes si nécessaire
    if skip_cols:
        skip = set(skip_cols)
        df_original = df_original.drop(columns=[c for c in df_original.columns if c in skip], errors="ignore")
        df_stationary = df_stationary.drop(columns=[c for c in df_stationary.columns if c in skip], errors="ignore")

    # --- 1) Long via helper commun
    df_long_orig = wide_to_long_timeseries(df_original, type_label="Original")
    df_long_stat = wide_to_long_timeseries(df_stationary, type_label="Stationnaire")

    df_plot = (
        pd.concat([df_long_orig, df_long_stat], ignore_index=True)
          .sort_values(["series", "type", "date"])
          .reset_index(drop=True)
    )

    # --- 2) Titres (méthode) via meta
    method_map, d2_series = {}, set()
    if meta is not None:
        for serie, info in meta.items():
            method = info.get("method")
            lags = int(info.get("lags", 0))

            if method == "logdiff":
                method_map[serie] = f"Δlog({lags})"
            elif method == "diff":
                method_map[serie] = f"Δ({lags})"
            elif method in ("none", "skip"):
                method_map[serie] = "niveau"
            else:
                method_map[serie] = "méthode inconnue"

            if lags == 2:
                d2_series.add(serie)

    def format_method_for_title(series: str) -> str:
        base = method_map.get(series, "méthode inconnue")
        if series in d2_series:
            return base.replace("Δ", "Δ²", 1)
        return base

    # --- 3) Plot (style cohérent avec plot_original_series)
    plt.style.use(style)
    sns.set_context("talk")

    g = sns.relplot(
        data=df_plot,
        x="date", y="value",
        row="series",
        col="type",
        col_order=["Original", "Stationnaire"],
        kind="line",
        linewidth=linewidth,
        facet_kws=dict(sharey=sharey),
        height=height,
        aspect=aspect,
        legend=False,
    )

    g.set_titles("")
    g.set_axis_labels("", "")

    # --- 4) Titre commun par ligne
    for i, serie in enumerate(g.row_names):
        left_ax, right_ax = g.axes[i, 0], g.axes[i, 1]
        method_txt = format_method_for_title(serie)
        row_title = f"{serie} (original / {method_txt})"

        left_pos, right_pos = left_ax.get_position(), right_ax.get_position()
        x_center = (left_pos.x0 + right_pos.x1) / 2
        y_top = left_pos.y1 + 0.02
        g.fig.text(x_center, y_top, row_title, ha="center", va="bottom", fontsize=11)

    # --- 5) Axes temps (même esprit que plot_original_series)
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    for ax in g.axes.flat:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis="x", labelrotation=0)  # cohérent avec plot_original_series
        ax.grid(False)

    plt.show()
    return df_plot