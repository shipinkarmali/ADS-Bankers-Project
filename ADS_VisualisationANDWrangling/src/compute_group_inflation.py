"""
Compute group-specific Laspeyres inflation for each archetype (tenure_type, income_quintile) using LCF expenditure shares and CPIH sub-indices.
"""

import pathlib
import warnings

import pandas as pd

from data_loaders import load_cpih_monthly, load_lcf_shares

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

CONCORDANCE = {
    "share_01_food_non_alcoholic":  "food_non_alcoholic",
    "share_02_alcohol_tobacco":     "alcohol_tobacco",
    "share_03_clothing_footwear":   "clothing_footwear",
    "share_04_actual_rent":         "actual_rents",
    "share_04_energy_other":        "non_rent_housing_fuel",
    "share_05_furnishings":         "furnishings",
    "share_06_health":              "health",
    "share_07_transport":           "transport",
    "share_08_communication":       "communication",
    "share_09_recreation_culture":  "recreation_culture",
    "share_10_education":           "education",
    "share_11_restaurants_hotels":  "restaurants_hotels",
    "share_12_misc_goods_services": "misc_goods_services",
}

ARCHETYPE_COLS = ["tenure_type", "income_quintile"]


# Takes monthly CPIH prices and computes annual price changes for each sub-index.
# Decomposes housing & fuel inflation into its rent and non-rent components using beta_rent as rent's weight.
def annual_price_changes(prices: pd.DataFrame) -> pd.DataFrame:
    cols = list(set(CONCORDANCE.values()) | {"housing_fuel_power", "actual_rents"})
    cols = [c for c in cols if c in prices.columns]
    annual = prices.groupby("fy_year")[cols].mean().sort_index()
    pct = (annual.pct_change() * 100).dropna().reset_index().rename(
        columns={"fy_year": "year"}
    )

    beta_rent = 0.26
    pct["non_rent_housing_fuel"] = (
        pct["housing_fuel_power"] - beta_rent * pct["actual_rents"]
    ) / (1 - beta_rent)

    return pct


def compute_archetype_shares(lcf: pd.DataFrame, arch_col: str) -> pd.DataFrame:
    """Return weighted mean COICOP share per (archetype, year)."""
    share_cols = list(CONCORDANCE.keys())
    rows = []
    for (grp, yr), sub in lcf.groupby([arch_col, "year"]):
        w = sub["household_weight"].fillna(0)
        total_weight = w.sum()
        if total_weight == 0:
            continue
        means = {}

        # For each category, compute the weighted mean.
        for c in share_cols:
            shares = sub[c].fillna(0)  
            weighted = shares * w       
            means[c] = float(weighted.sum() / total_weight)
        
        # Each row (archetype, year) has all the categories' weighted means.
        rows.append({"archetype_value": grp, "year": int(yr), **means})
    return pd.DataFrame(rows)

# shares is (archetype, year) -> {category: share} for each row.
# prices is (year) -> {category: price} for each row.
def laspeyres_inflation(shares: pd.DataFrame, prices: pd.DataFrame, arch_col: str) -> pd.DataFrame:
    """Apply lagged shares to next-year price changes and sum the contributions to get the group's Laspeyres inflation rate."""
    rows = []
    for _, shares_row in shares.iterrows():
        target_year = shares_row["year"] + 1
        price_row = prices[prices["year"] == target_year]
        if price_row.empty:
            continue
        price_row = price_row.iloc[0]
        category_contributions = []

        # LHS of CONCORDANCE is for shares df, RHS is for prices df.
        for share_col, price_col in CONCORDANCE.items():

            # Inflation contribution is the share of the category * the price change of the category.
            val = shares_row[share_col] * price_row[price_col]
            if pd.isna(val):
                continue
            category_contributions.append((price_col, val))

        # Category contributions is a list of tuples like ("food_non_alcoholic", 0.1234) so this fancy tuple unpacking sums the values.
        total = sum(v for _, v in category_contributions)
        
        for price_col, val in category_contributions:
            rows.append({
                "archetype_name": arch_col,
                "archetype_value": shares_row["archetype_value"],
                "year": int(target_year),
                "coicop_label": price_col,
                "contribution": val,
            })

        # Headline group inflation rate (sum of component contributions).
        rows.append({
            "archetype_name": arch_col,
            "archetype_value": shares_row["archetype_value"],
            "year": int(target_year),
            "coicop_label": "all_items",
            "contribution": total,
        })

    return pd.DataFrame(rows)


def main() -> None:
    lcf = load_lcf_shares()

    missing_lcf_cols = [c for c in CONCORDANCE.keys() if c not in lcf.columns]
    if missing_lcf_cols:
        raise ValueError(f"Missing share columns: {missing_lcf_cols}")

    prices = load_cpih_monthly()
    price_changes = annual_price_changes(prices)

    # Add a personal inflation proxy to every household.
    lcf["inflation_year"] = lcf["year"] + 1
    lcf["inflation_proxy"] = 0.0

    for share_col, price_col in CONCORDANCE.items():
        lookup = price_changes.set_index("year")[price_col]
        lcf["inflation_proxy"] += lcf[share_col] * lcf["inflation_year"].map(lookup)

    lcf = lcf.dropna(subset=["inflation_proxy"])
    lcf.to_csv(OUTPUT / "household_inflation.csv", index=False)

    # Group-level Laspeyres decomposition for each archetype dimension.
    archetypes_inflation = []
    for archetype in ARCHETYPE_COLS:
        if archetype not in lcf.columns:
            continue
        shares = compute_archetype_shares(lcf, archetype)
        archetypes_inflation.append(laspeyres_inflation(shares, price_changes, archetype))

    breakdown = pd.concat(archetypes_inflation, ignore_index=True)
    breakdown["archetype_value"] = breakdown["archetype_value"].astype(str)

    group_rates = (
        breakdown[breakdown["coicop_label"] == "all_items"]
        [["archetype_name", "archetype_value", "year", "contribution"]]
        .rename(columns={"contribution": "inflation_rate"})
    )
    group_rates.to_csv(OUTPUT / "group_inflation_rates.csv", index=False)
    breakdown.to_csv(OUTPUT / "inflation_decomposition.csv", index=False)


if __name__ == "__main__":
    main()
