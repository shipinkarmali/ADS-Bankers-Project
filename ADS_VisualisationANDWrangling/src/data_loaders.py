"""
data_loaders.py
- Loads manually-curated input data straight from Excel and Stata files.


Input files:
data/cleaned/MM23_cleaned.xlsx  – CPIH price indices (manually wrangled)
data/cleaned/HCI_cleaned.xlsx   – HCI validation data (manually wrangled)
data/output/lcf_expenditure_shares.csv – household shares (from wrangle_lcf)

Loader functions:
  load_cpih_monthly()       – monthly CPIH index panel (wide, one row per month)
  load_cpih_fy_indices()    – financial-year average CPIH indices
  load_hci_validation()     – HCI tenure index, long format, 2015+
  load_lcf_shares()         – LCF household expenditure shares (from CSV cache)
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLEANED = ROOT / "data" / "cleaned"
OUTPUT = ROOT / "data" / "output"

MM23_XLSX = CLEANED / "MM23_cleaned.xlsx"
HCI_XLSX = CLEANED / "HCI_cleaned.xlsx"

# LHS has Excel spreadsheet column names.
MM23_COLUMN_MAP = {
    "All Items":    "all_items",
    "Food":         "food_non_alcoholic",
    "Alcohol":      "alcohol_tobacco",
    "Clothing":     "clothing_footwear",
    "Housing":      "housing_fuel_power",
    "Furniture":    "furnishings",
    "Health":       "health",
    "Transport":    "transport",
    "Communication": "communication",
    "Recreation":   "recreation_culture",
    "Education":    "education",
    "Restaurants":  "restaurants_hotels",
    "Misc":         "misc_goods_services",
    "Actual Rent":  "actual_rents",
    "Energy":       "electricity_gas_fuels",
}

# HCI tenure short names -> official ONS HCI group names
HCI_TENURE_MAP = {
    "Mortgagor": "Mortgagor and other owner occupier",
    "Outright owner": "Outright owner occupier",
    "Private renter": "Private renter",
    "Social renter": "Social and other renter",
}


def load_cpih_monthly() -> pd.DataFrame:
    """
    Monthly CPIH indices in wide format (one row per month).

    Columns returned:
    date, actual_rents, alcohol_tobacco, all_items, clothing_footwear, communication, education, electricity_gas_fuels, food_non_alcoholic, furnishings, health, housing_fuel_power, misc_goods_services, recreation_culture, restaurants_hotels, transport,
    year, month, fy_year, fy
    """
    # Read the "Monthly" Excel spreadsheet that's manually wrangled.
    raw = pd.read_excel(MM23_XLSX, sheet_name="Monthly", header=None)

    # Skip header rows.
    data = raw.iloc[3:].reset_index(drop=True)

    # Map column positions to the official names (in the same order as Row 1 of the Financial Year Averages sheet).
    price_cols = [
        "all_items",          # col 3
        "food_non_alcoholic",  # col 4
        "alcohol_tobacco",     # col 5
        "clothing_footwear",   # col 6
        "housing_fuel_power",  # col 7
        "furnishings",         # col 8
        "health",              # col 9
        "transport",           # col 10
        "communication",       # col 11
        "recreation_culture",  # col 12
        "education",           # col 13
        "restaurants_hotels",  # col 14
        "misc_goods_services", # col 15
        "actual_rents",        # col 16
        "electricity_gas_fuels", # col 17
    ]

    # Build the output DataFrame
    out = pd.DataFrame()
    # Parse YYYYMM integer -> first-of-month timestamp
    yyyymm = pd.to_numeric(data.iloc[:, 0], errors="coerce")
    yyyymm = yyyymm.dropna().astype(int)
    out["date"] = pd.to_datetime(yyyymm.astype(str), format="%Y%m")

    # Copy the 15 price columns (aligning by the same rows we kept)
    valid_date_rows = pd.to_numeric(data.iloc[:, 0], errors="coerce").notna()
    for i, name in enumerate(price_cols, start=3):
        out[name] = pd.to_numeric(data.loc[valid_date_rows, i].values, errors="coerce")

    # Year / month / financial-year helpers (April–March FY)
    # fy_year is integer, fy is for human-readability.
    out["year"] = out["date"].dt.year.astype("int32")
    out["month"] = out["date"].dt.month.astype("int32")
    out["fy_year"] = np.where(out["month"] >= 4, out["year"], out["year"] - 1).astype("int32")
    out["fy"] = (out["fy_year"].astype(str) + "/" + (out["fy_year"] + 1).astype(str).str[-2:])

    return out.sort_values("date").reset_index(drop=True)


def load_cpih_fy_indices() -> pd.DataFrame:
    """
    Financial-year average CPIH indices.
    Returns one row per FY with one column per COICOP division, plus a two-digit FY label ('2015/16' etc.).
    """
    df = pd.read_excel(MM23_XLSX, sheet_name="Financial Year Averages")
    df = df.rename(columns={"Financial Year": "year"})
    df = df.rename(columns=MM23_COLUMN_MAP)
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df["fy"] = (df["year"].astype(str) + "/" + (df["year"] + 1).astype(str).str[-2:])
    # Re-order: 1. year, 2. fy, 3. then the 15 price columns
    cols = ["year", "fy"] + [c for c in df.columns if c not in ("year", "fy")]
    return df[cols].reset_index(drop=True)


def load_hci_validation() -> pd.DataFrame:
    """HCI tenure validation data (long format), filtered to 2015+.
    Columns returned:
    date, group, coicop_code, coicop_name, hci_price_index, metric, grouping, year, month, fy_year
    """
    wide = pd.read_excel(HCI_XLSX, sheet_name="HCI Tenure")
    wide = wide.dropna(subset=["Date"]).copy()
    wide["date"] = pd.to_datetime(wide["Date"], format="%b-%Y", errors="coerce")
    wide = wide.dropna(subset=["date"])

    # Convert HCI from wide to long format.
    long = wide.melt(
        id_vars=["date"],
        value_vars=list(HCI_TENURE_MAP.keys()),
        var_name="raw_tenure_label",
        value_name="hci_price_index",
    )

    long["group"] = long["raw_tenure_label"].map(HCI_TENURE_MAP)
    long = long.drop(columns=["raw_tenure_label"])
    long["hci_price_index"] = pd.to_numeric(long["hci_price_index"], errors="coerce")
    long = long.dropna(subset=["hci_price_index"])

    long["coicop_code"] = "0"
    long["coicop_name"] = "All items"
    long["metric"] = "index"
    long["grouping"] = "tenure"

    # Filter to 2015+ study period
    long = long[long["date"] >= "2015-01-01"].copy()

    long["year"] = long["date"].dt.year.astype("int32")
    long["month"] = long["date"].dt.month.astype("int32")
    long["fy_year"] = np.where(
        long["month"] >= 4, long["year"], long["year"] - 1
    ).astype("int32")

    return (
        long[["date", "group", "coicop_code", "coicop_name", "hci_price_index",
              "metric", "grouping", "year", "month", "fy_year"]]
        .sort_values(["grouping", "group", "date"])
        .reset_index(drop=True)
    )


def load_lcf_shares() -> pd.DataFrame:
    """Household-level LCF expenditure shares (and inflation proxy if the
    pipeline has been run), read from the CSV cache."""
    path = OUTPUT / "lcf_expenditure_shares.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"LCF expenditure shares CSV not found at {path}.\n"
            "Run `python src/run_pipeline.py` first to build it."
        )
    return pd.read_csv(path)


if __name__ == "__main__":
    # Self-test: load each dataset and print a quick summary.
    print("Loading CPIH monthly indices...")
    m = load_cpih_monthly()
    print(f"  {len(m):,} months, {m['date'].min():%Y-%m} to {m['date'].max():%Y-%m}")
    print(f"  Columns: {list(m.columns)}")

    print("\nLoading CPIH FY averages...")
    fy = load_cpih_fy_indices()
    print(f"  {len(fy):,} financial years, {fy['year'].min()}-{fy['year'].max()}")

    print("\nLoading HCI validation...")
    h = load_hci_validation()
    print(f"  {len(h):,} observations, {h['group'].nunique()} groups")
    print(f"  Date range: {h['date'].min():%Y-%m} to {h['date'].max():%Y-%m}")