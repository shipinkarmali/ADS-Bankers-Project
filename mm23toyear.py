import numpy as np
import pandas as pd
from pathlib import Path

mm23_clean_path = Path("cleaned/mm23_key_categories_yoy.csv")
out_path        = Path("cleaned/mm23_yearly_for_merge.csv")

print("Loading cleaned MM23...")
mm23 = pd.read_csv(mm23_clean_path)
mm23["month"] = pd.to_datetime(mm23["month"], errors="coerce")
mm23 = mm23.dropna(subset=["month"]).copy()
mm23["year"] = mm23["month"].dt.year

numeric_cols = [c for c in mm23.select_dtypes(include=[np.number]).columns if c != "year"]

mm23_yearly = (
    mm23.groupby("year")[numeric_cols]
    .mean()
    .reset_index()
    .sort_values("year")
    .reset_index(drop=True)
)

out_path.parent.mkdir(parents=True, exist_ok=True)
mm23_yearly.to_csv(out_path, index=False)
print(f"Saved to: {out_path}")
print(f"Final shape: {mm23_yearly.shape}")
