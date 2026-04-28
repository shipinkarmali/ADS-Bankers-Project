import numpy as np
import pandas as pd
from pathlib import Path

lcf_path        = Path("cleaned/lcf_expenditure_shares.csv")
mm23_yearly_path = Path("cleaned/mm23_yearly_for_merge.csv")
out_path        = Path("cleaned/household_inflation_dataset.csv")

share_mapping = {
    "share_01_food_non_alcoholic":  "cat_food",
    "share_02_alcohol_tobacco":     "cat_alcohol_tobacco",
    "share_03_clothing_footwear":   "cat_clothing",
    "share_04_housing_fuel_power":  "cat_housing",
    "share_05_furnishings":         "cat_furniture",
    "share_06_health":              "cat_health",
    "share_07_transport":           "cat_transport",
    "share_08_communication":       "cat_communication",
    "share_09_recreation_culture":  "cat_recreation",
    "share_10_education":           "cat_education",
    "share_11_restaurants_hotels":  "cat_restaurants",
    "share_12_misc_goods_services": "cat_misc",
}

print("Loading datasets...")
lcf  = pd.read_csv(lcf_path)
mm23 = pd.read_csv(mm23_yearly_path)
lcf.columns = lcf.columns.str.strip()
print(f"LCF shape: {lcf.shape}")
print(f"MM23 yearly shape: {mm23.shape}")

lcf["year"]  = pd.to_numeric(lcf["year"],  errors="coerce").astype(int)
mm23["year"] = pd.to_numeric(mm23["year"], errors="coerce").astype(int)

before = len(lcf)
mask = (
    (lcf["tenure_type"] == "unknown") |
    (lcf["hrp_age"] < 16) |
    (lcf["income_gross_weekly"] <= 0)
)
lcf = lcf[~mask].copy().reset_index(drop=True)
print(f"Removed {before - len(lcf)} rows, {len(lcf)} remaining")

lcf["inflation_year"] = lcf["year"] + 1

df = lcf.merge(
    mm23,
    left_on="inflation_year",
    right_on="year",
    how="left",
    suffixes=("", "_mm23"),
    validate="many_to_one",
    indicator=True
)

unmatched = sorted(df[df["_merge"] != "both"]["inflation_year"].dropna().unique())
if unmatched:
    print(f"Warning: no MM23 match for inflation years: {unmatched}")

df = df.drop(columns=["_merge"])
if "year_mm23" in df.columns:
    df = df.drop(columns=["year_mm23"])
elif "year_y" in df.columns:
    df = df.drop(columns=["year_y"])
    if "year_x" in df.columns:
        df = df.rename(columns={"year_x": "year"})

print(f"Merged shape: {df.shape}")

share_cols = list(share_mapping.keys())
cat_cols   = list(share_mapping.values())

df["share_sum_main_12"] = df[share_cols].sum(axis=1)

shares = df[share_cols].to_numpy(dtype=float)
infls  = df[cat_cols].to_numpy(dtype=float)

df["household_inflation_rate"] = np.sum(shares * infls, axis=1)

missing_mask = df[share_cols + cat_cols].isna().any(axis=1)
df.loc[missing_mask, "household_inflation_rate"] = np.nan

if "cpi_all_items_yoy_inflation" in df.columns:
    df["inflation_gap_vs_headline"] = df["household_inflation_rate"] - df["cpi_all_items_yoy_inflation"]

df["shock_period"] = np.where(df["inflation_year"].between(2021, 2023), 1, 0)

before = len(df)
df = df.dropna(subset=["household_inflation_rate"]).copy()
print(f"Dropped {before - len(df)} rows with missing household_inflation_rate")

print(f"\nFinal shape: {df.shape}")
print("\nhousehold_inflation_rate summary:")
print(df["household_inflation_rate"].describe().round(4).to_string())
print("\nMean by tenure_type:")
print(df.groupby("tenure_type")["household_inflation_rate"].mean().round(4).sort_values().to_string())

out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"\nSaved to: {out_path}")
