import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNetCV, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# elastic net pipeline, ols baseline + CV-picked alpha + forced low alpha
# feature set matches Model 2 from linear_regression_final.py:
#   3 tenure dummies (ref=own_outright) + 4 age-band dummies (ref=50_to_64)
#   + income_quintile (continuous) + 12 COICOP shares = 20 features total

csv_path = "cleaned/household_inflation_dataset.csv"
outdir = "elasticnet_household_outputs"
os.makedirs(outdir, exist_ok=True)

random_state = 42

year_col = "year"
weight_col = "household_weight"
raw_target_col = "household_inflation_rate"
target_col = "excess_household_inflation"

categorical_cols = [
    "tenure_type",
    "hrp_age_band",
]

# matches Model 2
numeric_cols = [
    "income_quintile",
    "share_01_food_non_alcoholic",
    "share_02_alcohol_tobacco",
    "share_03_clothing_footwear",
    "share_04_housing_fuel_power",
    "share_05_furnishings",
    "share_06_health",
    "share_07_transport",
    "share_08_communication",
    "share_09_recreation_culture",
    "share_10_education",
    "share_11_restaurants_hotels",
    "share_12_misc_goods_services",
]

tenure_ref = "own_outright"
age_ref = "50_to_64"

train_y = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
test_y = [2022, 2023]

l1_ratios = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
n_alphas = 100


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_data(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    needed = (
        [year_col, weight_col, raw_target_col]
        + categorical_cols
        + numeric_cols
    )
    df = df[needed].copy()

    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
    df[raw_target_col] = pd.to_numeric(df[raw_target_col], errors="coerce")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan})

    df = df.dropna(subset=[year_col, weight_col, raw_target_col]).copy()

    # weighted yearly mean -> excess inflation
    weighted_yearly = df[[year_col, raw_target_col, weight_col]].copy()
    weighted_yearly["weighted_target"] = weighted_yearly[raw_target_col] * weighted_yearly[weight_col]
    year_means = weighted_yearly.groupby(year_col, as_index=False).agg(
        weighted_sum=("weighted_target", "sum"),
        weight_sum=(weight_col, "sum"),
    )
    year_means["year_weighted_mean"] = year_means["weighted_sum"] / year_means["weight_sum"]
    year_means = year_means[[year_col, "year_weighted_mean"]]

    df = df.merge(year_means, on=year_col, how="left")
    df[target_col] = df[raw_target_col] - df["year_weighted_mean"]

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna("unknown")

    df_raw = df.copy()
    df_model = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=np.uint8)
    # drop explicit reference categories so Elastic Net uses the same reference cells as Model 2
    ref_cols = [f"tenure_type_{tenure_ref}", f"hrp_age_band_{age_ref}"]
    df_model = df_model.drop(columns=[c for c in ref_cols if c in df_model.columns], errors="ignore")
    return df_raw, df_model


def feature_cols(df):
    excluded = {target_col, raw_target_col, year_col, weight_col, "year_weighted_mean"}
    return [c for c in df.columns if c not in excluded]


def evaluate(y_true, y_pred):
    return {
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def features(feature):
    map = {
        "share_01_food_non_alcoholic": "Food Share","share_02_alcohol_tobacco": "Alcohol & Tobacco Share",
        "share_03_clothing_footwear": "Clothing Share","share_04_housing_fuel_power": "Housing & Fuel Share",
        "share_05_furnishings": "Furnishings Share","share_06_health": "Health Share",
        "share_07_transport": "Transport Share","share_08_communication": "Communication Share",
        "share_09_recreation_culture": "Recreation & Culture Share","share_10_education": "Education Share",
        "share_11_restaurants_hotels": "Restaurants & Hotels Share", "share_12_misc_goods_services": "Miscellaneous Goods & Services Share",
        "share_04_actual_rent": "Actual Rent Share","share_04_energy_other": "Energy Share",
        "household_size": "Household Size", "income_gross_weekly": "Gross Weekly Income",
        "income_equivalised": "Equivalised Income", "total_expenditure": "Total Expenditure", "hrp_age": "HRP Age",
    }
    if feature in map:
        return map[feature]
    for prefix, label in [("tenure_type_", "Tenure: "), ("income_quintile_", "Income Q"), ("hrp_age_band_", "Age: ")]:
        if feature.startswith(prefix):
            suffix = feature.replace(prefix, "").replace("_", " ").title().replace(".0", "")
            return label + suffix
    return feature.replace("_", " ").strip().title()


def main():
    print("running elastic net pipeline...")

    df_raw, df_model = load_data(csv_path)
    feat_cols = feature_cols(df_model)

    train_df = df_model[df_model[year_col].isin(train_y)].copy().reset_index(drop=True)
    test_df = df_model[df_model[year_col].isin(test_y)].copy().reset_index(drop=True)
    test_raw = df_raw[df_raw[year_col].isin(test_y)].copy().reset_index(drop=True)

    x_train = train_df[feat_cols].values
    y_train = train_df[target_col].values
    w_train = train_df[weight_col].values
    x_test = test_df[feat_cols].values
    y_test = test_df[target_col].values

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    all_metrics = {}

    # ols linear fit, no penalty
    ols = LinearRegression()
    ols.fit(x_train_scaled, y_train, sample_weight=w_train)
    y_pred_ols = ols.predict(x_test_scaled)
    ols_metrics = evaluate(y_test, y_pred_ols)
    ols_metrics["model"] = "OLS"
    ols_metrics["alpha"] = 0.0
    ols_metrics["l1_ratio"] = np.nan
    ols_metrics["n_nonzero"] = int(np.sum(ols.coef_ != 0))
    all_metrics["OLS"] = ols_metrics

    ols_coef_df = pd.DataFrame({
        "feature": feat_cols,
        "coefficient": ols.coef_,
        "abs_coefficient": np.abs(ols.coef_),
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    # elastic net let CV pick alpha + l1_ratio jointly
    elastic_cv = ElasticNetCV(
        l1_ratio=l1_ratios,
        alphas=np.logspace(-6, 2, n_alphas),
        cv=5,
        random_state=random_state,
        n_jobs=-1,
        max_iter=10000,
    )
    elastic_cv.fit(x_train_scaled, y_train, sample_weight=w_train)
    y_pred_cv = elastic_cv.predict(x_test_scaled)
    cv_metrics = evaluate(y_test, y_pred_cv)
    cv_metrics["model"] = "ElasticNet_CV"
    cv_metrics["alpha"] = float(elastic_cv.alpha_)
    cv_metrics["l1_ratio"] = float(elastic_cv.l1_ratio_)
    cv_metrics["n_nonzero"] = int(np.sum(elastic_cv.coef_ != 0))
    all_metrics["ElasticNet_CV"] = cv_metrics

    # low alpha override CV to keep some coefs, check sanity
    elastic_low = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=random_state, max_iter=10000)
    elastic_low.fit(x_train_scaled, y_train, sample_weight=w_train)
    y_pred_low = elastic_low.predict(x_test_scaled)
    low_metrics = evaluate(y_test, y_pred_low)
    low_metrics["model"] = "ElasticNet_low_alpha"
    low_metrics["alpha"] = 0.001
    low_metrics["l1_ratio"] = 0.5
    low_metrics["n_nonzero"] = int(np.sum(elastic_low.coef_ != 0))
    all_metrics["ElasticNet_low"] = low_metrics

    predictions_df = test_raw[[year_col, weight_col, raw_target_col, target_col]].copy()
    predictions_df["predicted_excess_inflation"] = y_pred_ols
    predictions_df["residual"] = predictions_df[target_col] - predictions_df["predicted_excess_inflation"]

    metrics_all = pd.DataFrame([all_metrics["OLS"], all_metrics["ElasticNet_CV"], all_metrics["ElasticNet_low"]])
    metrics_all.to_csv(os.path.join(outdir, "test_metrics.csv"), index=False)

    # single-row file consumed by ElasticNetOutputs.py
    ols_metrics_row = {k: v for k, v in ols_metrics.items()}
    ols_metrics_row["n_nonzero_coefs"] = ols_metrics_row.pop("n_nonzero")
    ols_metrics_row["n_total_coefs"] = len(feat_cols)
    pd.DataFrame([ols_metrics_row]).to_csv(os.path.join(outdir, "test_metrics_ols.csv"), index=False)

    predictions_df.to_csv(os.path.join(outdir, "predictions_test_years.csv"), index=False)
    ols_coef_df.to_csv(os.path.join(outdir, "coefficients.csv"), index=False)

    train_df.to_csv(os.path.join(outdir, "train_processed.csv"), index=False)
    test_df.to_csv(os.path.join(outdir, "test_processed.csv"), index=False)
    test_raw.to_csv(os.path.join(outdir, "test_raw.csv"), index=False)

    joblib.dump(ols, os.path.join(outdir, "elasticnet_model.joblib"))
    joblib.dump(scaler, os.path.join(outdir, "scaler.joblib"))
    joblib.dump(feat_cols, os.path.join(outdir, "feature_cols.joblib"))

    # summary of the numbers that we use in the report
    summary_path = os.path.join(outdir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Household Elastic Net\n\n")
        f.write(f"train years: {sorted(train_df[year_col].unique().tolist())}\n")
        f.write(f"test years : {sorted(test_df[year_col].unique().tolist())}\n")
        f.write(f"train rows : {len(train_df)}\n")
        f.write(f"test rows  : {len(test_df)}\n")
        f.write(f"features   : {len(feat_cols)}\n\n")

        f.write("Comparison: OLS vs Elastic Net (CV) vs Elastic Net (low alpha)\n")
        f.write(metrics_all[["model", "r2", "rmse", "alpha", "l1_ratio", "n_nonzero"]].to_string(index=False))
        f.write("\n\n")

        f.write("Top 15 OLS coefficients (standardised, by magnitude):\n")
        for _, row in ols_coef_df.head(15).iterrows():
            sign = "+" if row["coefficient"] > 0 else ""
            f.write(f"  {features(row['feature'])}: {sign}{row['coefficient']:.6f}\n")

    print(f"all outputs saved to {outdir}/")


if __name__ == "__main__":
    main()
