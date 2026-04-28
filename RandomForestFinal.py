import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

csv_path = "cleaned/household_inflation_dataset.csv"
outdir = "rf_household_outputs"
os.makedirs(outdir, exist_ok=True)

random_state = 42

year_col = "year"
weight_col = "household_weight"
raw_target_col = "household_inflation_rate"
target_col = "excess_household_inflation"

categorical_cols = [
    "tenure_type",
    "income_quintile",
    "hrp_age_band",
]

numeric_cols = [
    "household_size",
    "income_gross_weekly",
    "income_equivalised",
    "total_expenditure",
    "hrp_age",
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
    "share_04_actual_rent",
    "share_04_energy_other",
]

# swap these for the pre-crisis run: train=[2015..2019], test=[2020,2021]
train_years = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
test_years = [2022, 2023]

inner_splits = 3
tune_n_iter = 8

rf_fallback_params = dict(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def weighted_rmse(y_true, y_pred, weights):
    return float(np.sqrt(np.average((y_true - y_pred) ** 2, weights=weights)))


def weighted_mae(y_true, y_pred, weights):
    return float(np.average(np.abs(y_true - y_pred), weights=weights))


def load_data(path):
    print("Reading file from:", os.path.abspath(path))

    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    required_cols = (
        [year_col, weight_col, raw_target_col]
        + categorical_cols
        + numeric_cols
    )

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[required_cols].copy()

    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
    df[raw_target_col] = pd.to_numeric(df[raw_target_col], errors="coerce")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in categorical_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({"nan": np.nan, "None": np.nan})
        )

    df = df.dropna(subset=[year_col, weight_col, raw_target_col]).copy()

    tmp = df[[year_col, raw_target_col, weight_col]].copy()
    tmp["weighted_target"] = tmp[raw_target_col] * tmp[weight_col]

    year_weighted_means = (
        tmp.groupby(year_col, as_index=False)
        .agg(
            weighted_sum=("weighted_target", "sum"),
            weight_sum=(weight_col, "sum")
        )
    )
    year_weighted_means["year_weighted_mean"] = (
        year_weighted_means["weighted_sum"] / year_weighted_means["weight_sum"]
    )
    year_weighted_means = year_weighted_means[[year_col, "year_weighted_mean"]]

    df = df.merge(year_weighted_means, on=year_col, how="left")
    df[target_col] = df[raw_target_col] - df["year_weighted_mean"]

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna("unknown")

    df_raw = df.copy()

    df_model = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=False,
        dtype=np.uint8
    )

    return df_raw, df_model


def make_feature_cols(df):
    excluded = {
        target_col,
        raw_target_col,
        year_col,
        weight_col,
        "year_weighted_mean",
    }
    return [c for c in df.columns if c not in excluded]


def tune_rf(X_train, y_train, groups_train, weights_train):
    base = RandomForestRegressor(
        random_state=random_state,
        n_jobs=1
    )

    param_dist = {
        "n_estimators": [200, 400, 600],
        "max_depth": [8, 10, 12, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", 0.5],
    }

    n_groups = len(np.unique(groups_train))
    n_splits = min(inner_splits, n_groups)

    if n_splits < 2:
        model = RandomForestRegressor(
            **rf_fallback_params,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train, sample_weight=weights_train)
        return model, dict(rf_fallback_params), np.nan

    inner_cv = GroupKFold(n_splits=n_splits)

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=tune_n_iter,
        scoring="neg_root_mean_squared_error",
        cv=inner_cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    search.fit(
        X_train,
        y_train,
        groups=groups_train,
        sample_weight=weights_train
    )

    best_params = dict(search.best_params_)
    best_score = float(search.best_score_)

    best_model = RandomForestRegressor(
        **best_params,
        random_state=random_state,
        n_jobs=-1
    )
    best_model.fit(X_train, y_train, sample_weight=weights_train)

    return best_model, best_params, best_score


def run_train_test(df_raw, df_model):
    feature_cols = make_feature_cols(df_model)

    train_df = df_model[df_model[year_col].isin(train_years)].copy().reset_index(drop=True)
    test_df = df_model[df_model[year_col].isin(test_years)].copy().reset_index(drop=True)

    train_raw = df_raw[df_raw[year_col].isin(train_years)].copy().reset_index(drop=True)
    test_raw = df_raw[df_raw[year_col].isin(test_years)].copy().reset_index(drop=True)

    if train_df.empty or test_df.empty:
        raise ValueError("Train or test set is empty. Check train_years / test_years.")

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    g_train = train_df[year_col]
    w_train = train_df[weight_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    w_test = test_df[weight_col]

    print("HOUSEHOLD RANDOM FOREST PIPELINE")
    print(f"Train years: {sorted(train_df[year_col].unique().tolist())}")
    print(f"Test years : {sorted(test_df[year_col].unique().tolist())}")
    print(f"Train rows : {len(train_df)}")
    print(f"Test rows  : {len(test_df)}")
    print(f"Target     : {target_col}")
    print(f"Features   : {len(feature_cols)}")

    try:
        model, best_params, best_score = tune_rf(X_train, y_train, g_train, w_train)
        print(f"[tuning] Best params: {best_params}")
        print(f"[tuning] Best CV neg-RMSE: {best_score:.6f}")
    except Exception as e:
        print("[tuning] FAILED, using fallback params. Error:", repr(e))
        best_params = dict(rf_fallback_params)
        best_score = np.nan
        model = RandomForestRegressor(
            **best_params,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train, sample_weight=w_train)

    y_pred = model.predict(X_test)

    metrics = {
        "rmse": rmse(y_test, y_pred),
        "mae": mae(y_test, y_pred),
        "weighted_rmse": weighted_rmse(y_test.values, y_pred, w_test.values),
        "weighted_mae": weighted_mae(y_test.values, y_pred, w_test.values),
        "r2": float(r2_score(y_test, y_pred)),
        "train_n": len(train_df),
        "test_n": len(test_df),
        "inner_cv_neg_rmse": best_score if pd.notna(best_score) else np.nan,
    }

    predictions_df = test_raw[[year_col, weight_col, raw_target_col, target_col]].copy()
    predictions_df["predicted_excess_inflation"] = y_pred
    predictions_df["residual"] = predictions_df[target_col] - predictions_df["predicted_excess_inflation"]

    return (
        model,
        feature_cols,
        train_df,
        test_df,
        train_raw,
        test_raw,
        predictions_df,
        metrics,
        best_params,
    )


def main():
    df_raw, df_model = load_data(csv_path)

    (
        model,
        feature_cols,
        train_df,
        test_df,
        train_raw,
        test_raw,
        predictions_df,
        metrics,
        best_params,
    ) = run_train_test(df_raw, df_model)

    predictions_df.to_csv(os.path.join(outdir, "predictions_test_years.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(outdir, "test_metrics.csv"), index=False)
    pd.DataFrame([best_params]).to_csv(os.path.join(outdir, "best_params.csv"), index=False)

    train_df.to_csv(os.path.join(outdir, "train_processed.csv"), index=False)
    test_df.to_csv(os.path.join(outdir, "test_processed.csv"), index=False)
    train_raw.to_csv(os.path.join(outdir, "train_raw.csv"), index=False)
    test_raw.to_csv(os.path.join(outdir, "test_raw.csv"), index=False)

    joblib.dump(model, os.path.join(outdir, "rf_model.joblib"))
    joblib.dump(feature_cols, os.path.join(outdir, "feature_cols.joblib"))

    print("\nMetrics:")
    print(pd.DataFrame([metrics]).round(6))
    print(f"\nR² = {metrics['r2']:.4f}")
    print("\nBest params:")
    print(pd.DataFrame([best_params]))


if __name__ == "__main__":
    main()
