import numpy as np
import pandas as pd

def build_score_and_explainability_tables(
    *,
    bkt_score: pd.DataFrame,
    results_perm_mae_by_part: dict,
    results_perm_deviance_by_part: dict,
    results_shap_share_by_part: dict,
    models=None,
    unique_id_col: str = "unique_id",
    date_col: str = "ds",
    cutoff_col: str = "cutoff",
    target_col: str = "y",
    partition_col: str = "partition",
    top_k: int = 2,
    verbose: bool = True,
):
    """
    Construit :
    - score_df      : métriques de performance agrégées
    - score_df_exp  : score enrichi avec explicabilité
    - leaderboard_exp : top_k modèles par partition

    Paramètres
    ----------
    bkt_score : pd.DataFrame
        DataFrame de backtest contenant y, partitions, forecasts et intervalles.
    results_perm_mae_by_part : dict
        Dictionnaire {partition: {model: df_perm_mae}}
    results_perm_deviance_by_part : dict
        Dictionnaire {partition: {model: df_perm_dev}}
    results_shap_share_by_part : dict
        Dictionnaire {partition: {model: df_shap}}
    models : list[str] | None
        Liste des modèles à considérer. Ex: ["LR", "RIDGE", "LGBM"]
    unique_id_col, date_col, cutoff_col, target_col, partition_col : str
        Noms de colonnes.
    top_k : int
        Nombre de modèles à garder par partition dans le leaderboard.
    verbose : bool
        Affichage des résumés.

    Retour
    ------
    tuple
        long_sc2, score_df, score_df_exp, leaderboard_exp
    """

    if models is None:
        models = ["LR", "RIDGE", "LGBM"]

    tmp = bkt_score.copy()

    required_cols = [unique_id_col, date_col, target_col, partition_col]
    missing = [c for c in required_cols if c not in tmp.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans bkt_score: {missing}")

    # =========================================================
    # 1) Sécurité bornes (lower <= upper)
    # =========================================================
    for m in models:
        lo = f"{m}-lo-95"
        hi = f"{m}-hi-95"
        if lo in tmp.columns and hi in tmp.columns:
            tmp[[lo, hi]] = np.sort(tmp[[lo, hi]].to_numpy(), axis=1)

    # =========================================================
    # 2) Wide -> Long + features de scoring
    # =========================================================
    rows = []

    base_cols = [c for c in [unique_id_col, date_col, cutoff_col, target_col, partition_col] if c in tmp.columns]

    for m in models:
        lo = f"{m}-lo-95"
        hi = f"{m}-hi-95"

        if m not in tmp.columns:
            continue

        s = tmp[base_cols].copy()
        s["model_label"] = m
        s["model_name"] = m

        s["forecast"] = tmp[m]
        s["lower"] = tmp[lo] if lo in tmp.columns else np.nan
        s["upper"] = tmp[hi] if hi in tmp.columns else np.nan

        s["abs_err"] = (s[target_col] - s["forecast"]).abs()
        s["covered"] = ((s[target_col] >= s["lower"]) & (s[target_col] <= s["upper"])).astype(int)
        s["int_width"] = (s["upper"] - s["lower"]).abs()

        rows.append(s)

    if not rows:
        raise ValueError("Aucune ligne produite à partir des modèles demandés.")

    long_sc = pd.concat(rows, ignore_index=True)
    long_sc[partition_col] = long_sc[partition_col].astype(str)

    # =========================================================
    # 3) Ajouter partition ALL
    # =========================================================
    long_all = long_sc.copy()
    long_all[partition_col] = "ALL"
    long_sc2 = pd.concat([long_sc, long_all], ignore_index=True)

    # =========================================================
    # 4) Score agrégé
    # =========================================================
    score_df = (
        long_sc2
        .groupby([unique_id_col, "model_label", "model_name", partition_col], observed=True)
        .agg(
            mae=("abs_err", "mean"),
            coverage=("covered", "mean"),
            width=("int_width", "mean"),
            n=(target_col, "size"),
        )
        .reset_index()
    )

    # =========================================================
    # 5) Helpers explicabilité
    # =========================================================
    def extract_top1_perm_by_part(res_dict_by_part):
        rows_out = []

        for partition, models_dict in res_dict_by_part.items():
            if not isinstance(models_dict, dict):
                continue

            for model, df in models_dict.items():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                if "feature" not in df.columns or "ratio_mean" not in df.columns:
                    continue

                top = df.sort_values("ratio_mean", ascending=False).iloc[0]
                rows_out.append({
                    partition_col: str(partition),
                    "model_label": str(model),
                    "perm_top1_feature": str(top["feature"]),
                    "perm_top1_value": float(top["ratio_mean"]),
                })

        return pd.DataFrame(rows_out)

    def extract_top1_shap_by_part(res_dict_by_part):
        rows_out = []

        for partition, models_dict in res_dict_by_part.items():
            if not isinstance(models_dict, dict):
                continue

            for model, df in models_dict.items():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                if "feature" not in df.columns or "shap_share_mean" not in df.columns:
                    continue

                top = df.sort_values("shap_share_mean", ascending=False).iloc[0]
                rows_out.append({
                    partition_col: str(partition),
                    "model_label": str(model),
                    "shap_top1_feature": str(top["feature"]),
                    "shap_top1_value": float(top["shap_share_mean"]),
                })

        return pd.DataFrame(rows_out)

    perm_mae_top1 = extract_top1_perm_by_part(results_perm_mae_by_part).rename(columns={
        "perm_top1_feature": "perm_mae_top1_feature",
        "perm_top1_value": "perm_mae_top1",
    })

    perm_dev_top1 = extract_top1_perm_by_part(results_perm_deviance_by_part).rename(columns={
        "perm_top1_feature": "perm_dev_top1_feature",
        "perm_top1_value": "perm_dev_top1",
    })

    shap_top1 = extract_top1_shap_by_part(results_shap_share_by_part).rename(columns={
        "shap_top1_feature": "shap_share_top1_feature",
        "shap_top1_value": "shap_share_top1",
    })

    # =========================================================
    # 6) Ajouter ALL pondéré par n
    # =========================================================
    weights_df = (
        score_df[score_df[partition_col] != "ALL"][
            [partition_col, "model_label", "n"]
        ]
        .drop_duplicates()
    )

    def add_all_partition_weighted(df_exp, value_cols):
        if df_exp.empty:
            return df_exp

        dfw = df_exp.merge(weights_df, on=[partition_col, "model_label"], how="left")
        dfw["n"] = dfw["n"].fillna(0).astype(float)

        feature_cols = [c for c in dfw.columns if "feature" in c]

        out = []
        for model, g in dfw[dfw[partition_col] != "ALL"].groupby("model_label"):
            row = {"model_label": model, partition_col: "ALL"}

            if feature_cols:
                g2 = g.sort_values("n", ascending=False)
                for fc in feature_cols:
                    row[fc] = g2.iloc[0][fc]

            for vc in value_cols:
                num = (g[vc].astype(float) * g["n"]).sum()
                den = g["n"].sum()
                row[vc] = float(num / den) if den > 0 else np.nan

            out.append(row)

        df_all = pd.DataFrame(out)
        return pd.concat([df_exp, df_all], ignore_index=True)

    perm_mae_top1 = add_all_partition_weighted(perm_mae_top1, value_cols=["perm_mae_top1"])
    perm_dev_top1 = add_all_partition_weighted(perm_dev_top1, value_cols=["perm_dev_top1"])
    shap_top1 = add_all_partition_weighted(shap_top1, value_cols=["shap_share_top1"])

    # =========================================================
    # 7) Merge explicabilité
    # =========================================================
    score_df_exp = (
        score_df
        .merge(perm_mae_top1, on=[partition_col, "model_label"], how="left")
        .merge(perm_dev_top1, on=[partition_col, "model_label"], how="left")
        .merge(shap_top1, on=[partition_col, "model_label"], how="left")
    )

    # =========================================================
    # 8) Leaderboard enrichi
    # =========================================================
    leaderboard_exp = (
        score_df_exp
        .sort_values(
            by=[partition_col, "mae", "coverage", "width"],
            ascending=[True, True, False, True],
        )
        .groupby(partition_col, as_index=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    if verbose:
        print("\n=== SCORE COMPLET (Performance + Explicabilité) ===")
        print(score_df_exp.sort_values([partition_col, "mae"]).head(50))

        print(f"\n=== TOP {top_k} par partition (incluant ALL) ===")
        print(leaderboard_exp)

    return long_sc2, score_df, score_df_exp, leaderboard_exp
