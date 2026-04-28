import numpy as np
import pandas as pd
from pathlib import Path

mm23_path = Path("Data/mm23.csv")
out_path  = Path("cleaned/mm23_key_categories_yoy.csv")

mm23_wanted = {
    "CPI INDEX 00: ALL ITEMS 2015=100":                                 "cpi_all_items",
    "CPI INDEX 01 : FOOD AND NON-ALCOHOLIC BEVERAGES 2015=100":         "cat_food",
    "CPI INDEX 02:ALCOHOLIC BEVERAGES,TOBACCO & NARCOTICS 2015=100":    "cat_alcohol_tobacco",
    "CPI INDEX 03 : CLOTHING AND FOOTWEAR 2015=100":                    "cat_clothing",
    "CPI INDEX 04 : HOUSING, WATER AND FUELS 2015=100":                 "cat_housing",
    "CPI INDEX 05 : FURN, HH EQUIP & ROUTINE REPAIR OF HOUSE 2015=100": "cat_furniture",
    "CPI INDEX 06 : HEALTH 2015=100":                                   "cat_health",
    "CPI INDEX 07 : TRANSPORT 2015=100":                                "cat_transport",
    "CPI INDEX 08 : COMMUNICATION 2015=100":                            "cat_communication",
    "CPI INDEX 09 : RECREATION & CULTURE 2015=100":                     "cat_recreation",
    "CPI INDEX 10 : EDUCATION 2015=100":                                "cat_education",
    "CPI INDEX 11 : HOTELS, CAFES AND RESTAURANTS 2015=100":            "cat_restaurants",
    "CPI INDEX 12 : MISCELLANEOUS GOODS AND SERVICES 2015=100":         "cat_misc",
}

print("Loading raw MM23...")
mm23_raw = pd.read_csv(mm23_path, low_memory=False)
print(f"Raw MM23 shape: {mm23_raw.shape}")

title_str = mm23_raw["Title"].astype(str).str.strip().str.upper()
valid_mask = title_str.str.match(r"^\d{4}\s+[A-Z]{3}$")
mm23 = mm23_raw.loc[valid_mask].copy()
mm23["month"] = pd.to_datetime(title_str[valid_mask], format="%Y %b", errors="coerce")
mm23 = mm23.loc[mm23["month"].dt.year.between(2000, 2030)].copy()
print(f"Valid monthly rows: {len(mm23)}")

missing_cols = [c for c in mm23_wanted if c not in mm23.columns]
if missing_cols:
    raise ValueError(f"Missing MM23 columns: {missing_cols}")

mm23 = mm23[["month"] + list(mm23_wanted.keys())].copy()
mm23 = mm23.rename(columns=mm23_wanted)

value_cols = [c for c in mm23.columns if c != "month"]
for col in value_cols:
    mm23[col] = pd.to_numeric(mm23[col], errors="coerce")

mm23 = mm23.dropna(subset=value_cols, how="all").copy()
mm23 = mm23.sort_values("month").reset_index(drop=True)

for col in value_cols:
    mm23[f"{col}_yoy_inflation"] = 100 * (mm23[col] / mm23[col].shift(12) - 1)

mm23 = mm23.loc[mm23["cpi_all_items_yoy_inflation"].notna()].copy()
mm23 = mm23.loc[mm23["month"] >= "2005-01-01"].copy()
mm23 = mm23.sort_values("month").reset_index(drop=True)

out_path.parent.mkdir(parents=True, exist_ok=True)
mm23.to_csv(out_path, index=False)
print(f"Saved to: {out_path}")
print(f"Final shape: {mm23.shape}")
