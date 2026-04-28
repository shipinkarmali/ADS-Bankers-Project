import os
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score


outdir = "rf_household_outputs"

random_state = 42

year_col = "year"
weight_col = "household_weight"
raw_target_col = "household_inflation_rate"
target_col = "excess_household_inflation"

theme = {
    "navy": "#2E4F8A",
    "blue": "#4472C4",
    "light_blue": "#DCE6F2",
    "red": "#ED1C24",
    "light_red": "#FADBDC",
    "pink": "#4472C4",
    "light_pink": "#DCE6F2",
    "grey": "#6E6E6E",
    "light_grey": "#D9D9D9",
    "black": "#222222",
    "white": "#FFFFFF",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": theme["black"],
    "axes.labelcolor": theme["black"],
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.frameon": False,
    "grid.color": theme["light_grey"],
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "savefig.bbox": "tight",
})


def weighted_group_mean(x, weights):
    x = pd.Series(x)
    weights = pd.Series(weights)
    mask = x.notna() & weights.notna()
    if mask.sum() == 0:
        return np.nan
    return float(np.average(x[mask], weights=weights[mask]))


def feature_name(feature):
    mapping = {
        "share_01_food_non_alcoholic": "Food Share",
        "share_02_alcohol_tobacco": "Alcohol & Tobacco Share",
        "share_03_clothing_footwear": "Clothing Share",
        "share_04_housing_fuel_power": "Housing & Fuel Share",
        "share_05_furnishings": "Furnishings Share",
        "share_06_health": "Health Share",
        "share_07_transport": "Transport Share",
        "share_08_communication": "Communication Share",
        "share_09_recreation_culture": "Recreation & Culture Share",
        "share_10_education": "Education Share",
        "share_11_restaurants_hotels": "Restaurants & Hotels Share",
        "share_12_misc_goods_services": "Miscellaneous Goods & Services Share",
        "share_04_actual_rent": "Actual Rent Share",
        "share_04_energy_other": "Energy Share",
        "income_quintile": "Income Quintile",
        "hrp_age_band": "HRP Age Band",
        "tenure_type": "Tenure Type",
        "household_size": "Household Size",
        "income_gross_weekly": "Gross Weekly Income",
        "income_equivalised": "Equivalised Income",
        "total_expenditure": "Total Expenditure",
        "hrp_age": "HRP Age",
    }

    if feature in mapping:
        return mapping[feature]

    if feature.startswith("tenure_type_"):
        return "Tenure: " + feature.replace("tenure_type_", "").replace("_", " ").title()

    if feature.startswith("income_quintile_"):
        return "Income Quintile: " + feature.replace("income_quintile_", "").replace("_", " ").title()

    if feature.startswith("hrp_age_band_"):
        return "HRP Age Band: " + feature.replace("hrp_age_band_", "").replace("_", " ").title()

    cleaned = feature.replace("_", " ").strip().title()

    replacements = {
        "Hrp": "HRP",
        "Yoy": "YoY",
        "Cpi": "CPI",
        "And": "&",
    }

    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)

    return cleaned


def get_r2_from_metrics_or_compute(metrics_df, y_true, y_pred):
    # be flexible about the column name in case the metrics schema drifts
    possible_cols = ["r2", "R2", "test_r2", "R_squared", "r_squared"]
    for col in possible_cols:
        if col in metrics_df.columns:
            return float(metrics_df[col].iloc[0])
    return float(r2_score(y_true, y_pred))


def load_saved_objects():
    model = joblib.load(os.path.join(outdir, "rf_model.joblib"))
    feature_cols = joblib.load(os.path.join(outdir, "feature_cols.joblib"))

    test_df = pd.read_csv(os.path.join(outdir, "test_processed.csv"))
    test_raw = pd.read_csv(os.path.join(outdir, "test_raw.csv"))
    predictions_df = pd.read_csv(os.path.join(outdir, "predictions_test_years.csv"))
    metrics_df = pd.read_csv(os.path.join(outdir, "test_metrics.csv"))
    best_params_df = pd.read_csv(os.path.join(outdir, "best_params.csv"))

    return model, feature_cols, test_df, test_raw, predictions_df, metrics_df, best_params_df


def compute_importance(model, X_test, y_test, feature_cols):
    rf_importance_df = pd.DataFrame({
        "feature": feature_cols,
        "rf_importance": model.feature_importances_,
    }).sort_values("rf_importance", ascending=False).reset_index(drop=True)

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=random_state,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error"
    )

    perm_importance_df = pd.DataFrame({
        "feature": feature_cols,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std": perm.importances_std,
    }).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)

    return rf_importance_df, perm_importance_df


def make_group_tables(test_raw, predictions_df):
    temp = test_raw.copy().reset_index(drop=True)
    preds = predictions_df[["predicted_excess_inflation"]].copy().reset_index(drop=True)
    temp = pd.concat([temp, preds], axis=1)

    def summarise_by(group_cols):
        rows = []
        grouped = temp.groupby(group_cols, dropna=False)

        for keys, g in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)

            row = {}
            for col, key in zip(group_cols, keys):
                row[col] = key

            row["n"] = len(g)
            row["mean_actual_excess"] = weighted_group_mean(g[target_col], g[weight_col])
            row["mean_predicted_excess"] = weighted_group_mean(g["predicted_excess_inflation"], g[weight_col])
            row["mean_actual_household_inflation"] = weighted_group_mean(g[raw_target_col], g[weight_col])

            rows.append(row)

        return pd.DataFrame(rows)

    by_tenure = summarise_by(["tenure_type"]).sort_values("mean_actual_excess", ascending=False).reset_index(drop=True)
    by_income = summarise_by(["income_quintile"]).reset_index(drop=True)
    by_age = summarise_by(["hrp_age_band"]).sort_values("mean_actual_excess", ascending=False).reset_index(drop=True)
    tenure_income = summarise_by(["tenure_type", "income_quintile"]).sort_values("mean_actual_excess", ascending=False).reset_index(drop=True)

    return by_tenure, by_income, by_age, tenure_income


def make_plots(predictions_df, metrics_df, rf_importance_df, perm_importance_df, by_tenure):
    y_true = predictions_df[target_col]
    y_pred = predictions_df["predicted_excess_inflation"]

    r2 = get_r2_from_metrics_or_compute(metrics_df, y_true, y_pred)

    # fig 1 - actual vs predicted as a hexbin density plot
    plt.figure(figsize=(7.5, 7))
    hb = plt.hexbin(
        y_true,
        y_pred,
        gridsize=45,
        cmap="Blues",
        mincnt=1,
        linewidths=0.0,
    )

    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--", color=theme["grey"],
             linewidth=1.5, label="Perfect fit")

    plt.xlabel("Actual excess household inflation (pp)")
    plt.ylabel("Predicted excess household inflation (pp)")
    plt.title("Model predictions vs actual excess inflation\n(2022–2023 test set)")
    plt.grid(True, axis="both", alpha=0.35)

    cbar = plt.colorbar(hb)
    cbar.set_label("Number of households")
    cbar.outline.set_visible(False)

    plt.text(
        0.04, 0.96,
        rf"$R^2 = {r2:.3f}$",
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=theme["light_blue"], edgecolor=theme["navy"])
    )
    plt.legend(loc="lower right", frameon=False)

    out = os.path.join(outdir, "fig1_predicted_vs_actual.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")

    # fig 2 - built-in RF importance (gini)
    rf_plot = rf_importance_df.head(15).copy()
    rf_plot["feature_pretty"] = rf_plot["feature"].apply(feature_name)
    rf_plot = rf_plot.sort_values("rf_importance", ascending=True)

    plt.figure(figsize=(10, 6.5))
    plt.barh(
        rf_plot["feature_pretty"],
        rf_plot["rf_importance"],
        color=theme["blue"],
        edgecolor=theme["navy"],
        alpha=0.95,
    )
    plt.xlabel("Random Forest feature importance")
    plt.title("Top 15 Random Forest feature importances")
    plt.grid(True, axis="x", alpha=0.35)

    out = os.path.join(outdir, "fig2_rf_importance.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")

    # fig 3 - permutation importance (scaled x1000 to make small values readable)
    perm_plot = perm_importance_df.head(15).copy()
    perm_plot["feature_pretty"] = perm_plot["feature"].apply(feature_name)
    perm_plot["perm_importance_scaled"] = perm_plot["perm_importance_mean"] * 1000
    perm_plot = perm_plot.sort_values("perm_importance_scaled", ascending=True)

    plt.figure(figsize=(10, 6.5))
    plt.barh(
        perm_plot["feature_pretty"],
        perm_plot["perm_importance_scaled"],
        color=theme["pink"],
        edgecolor=theme["navy"],
        alpha=0.95,
    )
    plt.xlabel(r"Permutation importance ($\times 10^{-3}$)")
    plt.title("Top 15 permutation importances")
    plt.grid(True, axis="x", alpha=0.35)

    out = os.path.join(outdir, "fig3_permutation_importance.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")

    # fig 4 - mean excess inflation by tenure
    by_tenure_plot = by_tenure.copy()

    tenure_label_map = {
        "owned_outright": "Owned outright",
        "owned with mortgage": "Owned with mortgage",
        "owned_with_mortgage": "Owned with mortgage",
        "social_rented": "Social rented",
        "private_rented": "Private rented",
        "private rented": "Private rented",
        "social rented": "Social rented",
    }

    by_tenure_plot["tenure_pretty"] = by_tenure_plot["tenure_type"].replace(tenure_label_map)

    plt.figure(figsize=(8.5, 4.8))
    plt.bar(
        by_tenure_plot["tenure_pretty"],
        by_tenure_plot["mean_actual_excess"],
        color=theme["pink"],
        edgecolor=theme["navy"],
        alpha=0.95,
    )
    plt.axhline(0, color=theme["black"], linewidth=1.1)
    plt.ylabel("Mean excess household inflation")
    plt.title("Mean excess household inflation by tenure type")
    plt.xticks(rotation=18, ha="right")
    plt.grid(True, axis="y", alpha=0.35)

    out = os.path.join(outdir, "fig4_mean_excess_by_tenure.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


def plot_permutation_importance_with_errorbars(perm_importance_df):
    # same as fig 3 but with error bars from the permutation repeats
    perm_plot = perm_importance_df.head(15).copy()
    perm_plot["feature_pretty"] = perm_plot["feature"].apply(feature_name)
    perm_plot["perm_importance_scaled"] = perm_plot["perm_importance_mean"] * 1000
    perm_plot["perm_importance_std_scaled"] = perm_plot["perm_importance_std"] * 1000
    perm_plot = perm_plot.sort_values("perm_importance_scaled", ascending=True)

    plt.figure(figsize=(10.5, 6.8))
    plt.barh(
        perm_plot["feature_pretty"],
        perm_plot["perm_importance_scaled"],
        xerr=perm_plot["perm_importance_std_scaled"],
        color=theme["light_blue"],
        edgecolor=theme["navy"],
        alpha=0.95,
        error_kw={"ecolor": theme["navy"], "elinewidth": 1.2, "capsize": 3},
    )
    plt.xlabel(r"Permutation importance ($\times 10^{-3}$)")
    plt.title("Top 15 permutation importances with error bars")
    plt.grid(True, axis="x", alpha=0.35)

    out = os.path.join(outdir, "fig5_permutation_importance_errorbars.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


def save_summary(metrics_df, best_params_df, perm_importance_df, by_tenure, by_income, by_age):
    out = os.path.join(outdir, "summary.txt")

    metrics = metrics_df.iloc[0].to_dict()
    best_params = best_params_df.iloc[0].to_dict()

    with open(out, "w", encoding="utf-8") as f:
        f.write("=== HOUSEHOLD RANDOM FOREST ===\n\n")

        f.write("Performance on held-out years:\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

        f.write("\nBest parameters:\n")
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")

        f.write("\nTop 15 permutation importance features:\n")
        for _, row in perm_importance_df.head(15).iterrows():
            f.write(f"{feature_name(row['feature'])}: {row['perm_importance_mean']:.6f}\n")

        f.write("\nMean excess inflation by tenure type:\n")
        for _, row in by_tenure.iterrows():
            f.write(
                f"{row['tenure_type']}: "
                f"actual={row['mean_actual_excess']:.6f}, "
                f"predicted={row['mean_predicted_excess']:.6f}\n"
            )

        f.write("\nMean excess inflation by income quintile:\n")
        for _, row in by_income.iterrows():
            f.write(
                f"{row['income_quintile']}: "
                f"actual={row['mean_actual_excess']:.6f}, "
                f"predicted={row['mean_predicted_excess']:.6f}\n"
            )

        f.write("\nMean excess inflation by age band:\n")
        for _, row in by_age.iterrows():
            f.write(
                f"{row['hrp_age_band']}: "
                f"actual={row['mean_actual_excess']:.6f}, "
                f"predicted={row['mean_predicted_excess']:.6f}\n"
            )

    print(f"Saved: {out}")


def main():
    model, feature_cols, test_df, test_raw, predictions_df, metrics_df, best_params_df = load_saved_objects()

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    rf_importance_df, perm_importance_df = compute_importance(
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_cols=feature_cols
    )

    by_tenure, by_income, by_age, tenure_income = make_group_tables(test_raw, predictions_df)

    rf_importance_df.to_csv(os.path.join(outdir, "rf_importance.csv"), index=False)
    perm_importance_df.to_csv(os.path.join(outdir, "permutation_importance.csv"), index=False)
    by_tenure.to_csv(os.path.join(outdir, "group_results_by_tenure.csv"), index=False)
    by_income.to_csv(os.path.join(outdir, "group_results_by_income.csv"), index=False)
    by_age.to_csv(os.path.join(outdir, "group_results_by_age_band.csv"), index=False)
    tenure_income.to_csv(os.path.join(outdir, "group_results_tenure_income.csv"), index=False)

    print("savedtables")

    make_plots(predictions_df, metrics_df, rf_importance_df, perm_importance_df, by_tenure)
    plot_permutation_importance_with_errorbars(perm_importance_df)
    save_summary(metrics_df, best_params_df, perm_importance_df, by_tenure, by_income, by_age)

    print("finished")


if __name__ == "__main__":
    main()
