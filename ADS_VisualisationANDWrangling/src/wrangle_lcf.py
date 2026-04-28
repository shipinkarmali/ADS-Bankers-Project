"""
wrangle_lcf.py
==============
Build a harmonised panel of LCF household expenditure shares and archetype
flags for financial years 2015/16 - 2023/24.

This is a linear top-to-bottom script: read each year's Stata files, combine,
clean, compute shares, tag archetypes, save one CSV.

Outputs one file:
    data/output/lcf_expenditure_shares.csv

Two archetype dimensions are produced:
    tenure_type, income_quintile
"""

import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "LCF"
OUTPUT = ROOT / "data" / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Stata file paths, one dvhh per year
LCF_DVHH = {
    2015: RAW / "LCF_2015/stata/stata11_se/2015-16_dvhh_ukanon.dta",
    2016: RAW / "LCF_2016/stata/stata13_se/2016_17_dvhh_ukanon.dta",
    2017: RAW / "LCF_2017/stata/stata11_se/dvhh_ukanon_2017-18.dta",
    2018: RAW / "LCF_2018/stata/stata13/2018_dvhh_ukanon.dta",
    2019: RAW / "LCF_2019/stata/stata13/lcfs_2019_dvhh_ukanon.dta",
    2020: RAW / "LCF_2020/stata/stata13/lcfs_2020_dvhh_ukanon.dta",
    2021: RAW / "LCF_2021/stata/stata13_se/lcfs_2021_dvhh_ukanon.dta",
    2022: RAW / "LCF_2022/stata/stata13_se/dvhh_ukanon_2022.dta",
    2023: RAW / "LCF_2023/stata/stata13_se/dvhh_ukanon_v2_2023.dta",
}

# COICOP division expenditure columns (weekly £).  p600t is overall total.
COICOP_COLS = [f"p60{d}t" if d < 10 else f"p6{d}t" for d in range(1, 13)]
COICOP_LABELS = {
    "p601t": "01_food_non_alcoholic",
    "p602t": "02_alcohol_tobacco",
    "p603t": "03_clothing_footwear",
    "p604t": "04_housing_fuel_power",
    "p605t": "05_furnishings",
    "p606t": "06_health",
    "p607t": "07_transport",
    "p608t": "08_communication",
    "p609t": "09_recreation_culture",
    "p610t": "10_education",
    "p611t": "11_restaurants_hotels",
    "p612t": "12_misc_goods_services",
}

# Household demographic/identifier columns to keep + rename.
DVHH_COLS = {
    "case": "household_id",
    "weighta": "household_weight",
    "a049": "household_size",
    "a121": "tenure_code",
    "p389p": "income_gross_weekly",
    "eqincdmp": "income_equivalised",
    "p600t": "total_expenditure",
    "b010": "rent_weekly",
}

TENURE_MAP = {
    1: "social_rent",    2: "social_rent",
    3: "private_rent",   4: "private_rent",
    5: "own_outright",
    6: "own_mortgage",   7: "own_mortgage",
    # 8 = rent free; left unmapped (too few, ~50/year)
}

# Plausibility bounds on weekly household expenditure (£)
EXP_LOW, EXP_HIGH = 30.0, 3000.0


def _read_stata(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_stata(path, convert_categoricals=False)
    df.columns = df.columns.str.lower()
    return df


# Load Stata files per year
years = sorted(LCF_DVHH.keys())
dvhh_frames = []
for yr in years:
    hh = _read_stata(LCF_DVHH[yr])
    keep = [c for c in (list(DVHH_COLS.keys()) + COICOP_COLS) if c in hh.columns]
    hh = hh[keep].rename(columns=DVHH_COLS)
    hh["year"] = yr
    dvhh_frames.append(hh)

df = pd.concat(dvhh_frames, ignore_index=True)


# Clean expenditure: negatives become NaN, then apply our plausibility filter
df.loc[df["total_expenditure"] < 0, "total_expenditure"] = np.nan
for col in COICOP_COLS:
    if col in df.columns:
        df.loc[df[col] < 0, col] = np.nan

df = df[df["total_expenditure"].between(EXP_LOW, EXP_HIGH, inclusive="both")].copy()

# Denominator consistency: if the LCF total differs from the sum of COICOP
# divisions by >1%, use the division sum to keep shares internally consistent.
coicop_sum = df[COICOP_COLS].sum(axis=1)
both_pos = (df["total_expenditure"] > 0) & (coicop_sum > 0)
ratio = df.loc[both_pos, "total_expenditure"] / coicop_sum[both_pos]
n_bad = ((ratio < 0.99) | (ratio > 1.01)).sum()
if n_bad > 0:
    denom = coicop_sum.replace(0, np.nan)
else:
    denom = df["total_expenditure"].replace(0, np.nan)


# Compute COICOP expenditure shares
for col, label in COICOP_LABELS.items():
    df[f"share_{label}"] = df[col] / denom

# Split COICOP 04 into actual rent vs energy + other (needed for inflation calc)
rent = df["rent_weekly"].fillna(0).clip(lower=0)
p604 = df["p604t"].fillna(0).clip(lower=0)
rent_capped = rent.clip(upper=p604)
df["share_04_actual_rent"] = rent_capped / denom
df["share_04_energy_other"] = (p604 - rent_capped).clip(lower=0) / denom

# Domain-based household filter: incomplete/erroneous diaries.
# (winsorisation would break compositional constraint)
exclude = (
    (df["share_01_food_non_alcoholic"] == 0) |
    (df["share_04_housing_fuel_power"] == 0)
)
df = df[~exclude].copy()


# Build archetypes
df["tenure_type"] = df["tenure_code"].map(TENURE_MAP).fillna("unknown")

# Weighted income quintiles (modified-OECD equivalised income) within year
df.loc[df["income_equivalised"] <= 0, "income_equivalised"] = np.nan
df["income_quintile"] = np.nan
for yr in df["year"].unique():
    mask = (df["year"] == yr) & df["income_equivalised"].notna()
    sub = df.loc[mask, ["income_equivalised", "household_weight"]].sort_values(
        "income_equivalised"
    )
    cum_share = sub["household_weight"].fillna(1).cumsum() / \
        sub["household_weight"].fillna(1).sum()
    thresholds = np.arange(1, 6) / 5.0
    q = np.clip(np.searchsorted(thresholds, cum_share.values, side="left") + 1, 1, 5)
    df.loc[sub.index, "income_quintile"] = q.astype(float)


# Drop internal columns and save
internal_cols = COICOP_COLS + ["tenure_code", "rent_weekly"]
df = df.drop(columns=[c for c in internal_cols if c in df.columns])
df.to_csv(OUTPUT / "lcf_expenditure_shares.csv", index=False)
