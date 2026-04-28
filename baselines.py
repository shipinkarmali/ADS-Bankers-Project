import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data_file  = "cleaned/household_inflation_dataset.csv"
output_dir = "outputs/baseline_outputs"
os.makedirs(output_dir, exist_ok=True)

year_col       = "year"
weight_col     = "household_weight"
raw_target_col = "household_inflation_rate"
target_col     = "excess_household_inflation"

train_years = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
test_years  = [2022, 2023]


def weighted_mean(values, weights):
    return float(np.average(values, weights=weights))


def weighted_mean_by_group(df, group_cols, target, weight):
    """Return {group_key: weighted_mean(target)}; key is a scalar for one column, tuple for many."""
    lookup = {}
    is_single_col = isinstance(group_cols, str) or len(group_cols) == 1
    cols = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    for key, group in df.groupby(cols, dropna=False):
        # pandas wraps single-column keys in a 1-tuple; unwrap for convenience
        if is_single_col and isinstance(key, tuple):
            key = key[0]
        lookup[key] = weighted_mean(group[target].values, group[weight].values)
    return lookup


def report(name, y_true, y_pred, weights):
    y_true  = np.asarray(y_true,  dtype=float)
    y_pred  = np.asarray(y_pred,  dtype=float)
    weights = np.asarray(weights, dtype=float)

    r2   = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))

    # weighted versions, for reference only
    weighted_rmse = float(np.sqrt(np.average((y_true - y_pred) ** 2, weights=weights)))
    weighted_mae  = float(np.average(np.abs(y_true - y_pred),         weights=weights))

    print(f"{name:38s}  R2={r2:+.4f}   RMSE={rmse:.4f}   MAE={mae:.4f}   "
          f"wRMSE={weighted_rmse:.4f}   wMAE={weighted_mae:.4f}")
    return dict(model=name, r2=r2, rmse=rmse, mae=mae,
                weighted_rmse=weighted_rmse, weighted_mae=weighted_mae)


def main():
    print("=" * 78)
    print("Baselines excess household inflation (2022–2023 test set)")
    print("=" * 78)

    df = pd.read_csv(data_file, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=[year_col, weight_col, raw_target_col]).copy()

    # Build the excess-inflation target: each household minus its year's weighted mean
    weighted = df[[year_col, raw_target_col, weight_col]].copy()
    weighted["weighted_target"] = weighted[raw_target_col] * weighted[weight_col]
    year_means = weighted.groupby(year_col, as_index=False).agg(
        weighted_sum=("weighted_target", "sum"),
        weight_sum  =(weight_col, "sum"))
    year_means["year_weighted_mean"] = year_means["weighted_sum"] / year_means["weight_sum"]
    df = df.merge(year_means[[year_col, "year_weighted_mean"]], on=year_col, how="left")
    df[target_col] = df[raw_target_col] - df["year_weighted_mean"]

    train = df[df[year_col].isin(train_years)].copy()
    test  = df[df[year_col].isin(test_years )].copy()

    # drop test rows with unknown tenure
    test = test[test["tenure_type"].notna()].copy()

    print(f"Train rows (2015-2021): {len(train):,}")
    print(f"Test  rows (2022-2023): {len(test):,}")
    print(f"Target mean (train, weighted): "
          f"{weighted_mean(train[target_col], train[weight_col]):+.4f}")
    print(f"Target mean (test , weighted): "
          f"{weighted_mean(test [target_col], test [weight_col]):+.4f}")
    print()

    results = []

    # Base0 - predict the training weighted mean for everyone
    train_mean = weighted_mean(train[target_col], train[weight_col])
    print("\n--- TRAIN-SET evaluation ---")
    report("B0 (train)", train[target_col],
           np.full(len(train), train_mean), train[weight_col])
    print("--- TEST-SET evaluation ---")
    results.append(report("B0: constant (train wmean)", test[target_col],
                          np.full(len(test), train_mean), test[weight_col]))

    # Base1 - predict the tenure-group training mean
    tenure_means = weighted_mean_by_group(train, ["tenure_type"], target_col, weight_col)
    print("\nTenure-group training means:")
    for tenure, mean in sorted(tenure_means.items()):
        print(f"  {tenure:15s}  {mean:+.4f}")
    print("\n--- TRAIN-SET evaluation ---")
    train_pred_b1 = train["tenure_type"].map(lambda t: tenure_means.get(t, train_mean)).values
    report("B1 (train)", train[target_col], train_pred_b1, train[weight_col])
    print("--- TEST-SET evaluation ---")
    test_pred_b1 = test.apply(lambda r: tenure_means.get(r["tenure_type"], train_mean), axis=1).values
    results.append(report("B1: tenure-group mean", test[target_col],
                          test_pred_b1, test[weight_col]))

    # Base2 - predict the tenure x income-quintile training mean (with fallbacks)
    train_iq = train.copy()
    test_iq  = test.copy()
    train_iq["iq"] = pd.to_numeric(train_iq["income_quintile"], errors="coerce")
    test_iq ["iq"] = pd.to_numeric(test_iq ["income_quintile"], errors="coerce")
    train_iq_fit = train_iq.dropna(subset=["iq"])
    # test rows with unknown quintile are kept; they fall back to tenure mean or global mean
    tenure_income_means = weighted_mean_by_group(train_iq_fit, ["tenure_type", "iq"],
                                                 target_col, weight_col)

    def predict_b2(row):
        key = (row["tenure_type"], row["iq"])
        if key in tenure_income_means:
            return tenure_income_means[key]
        if row["tenure_type"] in tenure_means:
            return tenure_means[row["tenure_type"]]
        return train_mean

    print("\n--- TRAIN-SET evaluation ---")
    train_pred_b2 = train_iq.apply(predict_b2, axis=1).values
    report("B2 (train)", train_iq[target_col], train_pred_b2, train_iq[weight_col])
    print("--- TEST-SET evaluation ---")
    test_pred_b2 = test_iq.apply(predict_b2, axis=1).values
    results.append(report("B2: tenure x income-quintile mean",
                          test_iq[target_col], test_pred_b2, test_iq[weight_col]))

    pd.DataFrame(results).to_csv(os.path.join(output_dir, "baseline_metrics.csv"), index=False)
    print(f"\nSaved: {output_dir}/baseline_metrics.csv")
    print("\nCompare against:")
    print("  Model 1  (demographics only):     R2 = -0.053")
    print("  Model 2  (+ shares)              R2 = +0.082")
    print("  Elastic Net (CV):                R2 = +0.085")
    print("  Random Forest:                   R2 = +0.062")
    print("  NN full (33 features):           R2 = +0.11")
    print("  NN top-6:                        R2 = +0.28")


if __name__ == "__main__":
    main()
