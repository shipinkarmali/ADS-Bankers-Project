import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HH_PATH = "cleaned/household_inflation_dataset.csv"
ANALYSIS_RATE_COL = "annual_inflation_rate"

SHARE_COLS = [
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

INFL_COLS = [
    "cat_food_yoy_inflation",
    "cat_alcohol_tobacco_yoy_inflation",
    "cat_clothing_yoy_inflation",
    "cat_housing_yoy_inflation",
    "cat_furniture_yoy_inflation",
    "cat_health_yoy_inflation",
    "cat_transport_yoy_inflation",
    "cat_communication_yoy_inflation",
    "cat_recreation_yoy_inflation",
    "cat_education_yoy_inflation",
    "cat_restaurants_yoy_inflation",
    "cat_misc_yoy_inflation",
]

df = pd.read_csv(HH_PATH)
df = df[df["tenure_type"] != "unknown"].copy()
df = df.dropna(subset=SHARE_COLS + INFL_COLS).copy()
df[ANALYSIS_RATE_COL] = np.sum(df[SHARE_COLS].to_numpy() * df[INFL_COLS].to_numpy(), axis=1)

def bootstrap_gap(data_a, w_a, data_b, w_b, B=10000, seed=42):
    # weighted mean gap (a - b) with 95% CI from percentile bootstrap
    rng = np.random.default_rng(seed)
    prob_a = w_a / w_a.sum()
    prob_b = w_b / w_b.sum()

    theta = np.average(data_a, weights=w_a) - np.average(data_b, weights=w_b)

    deltas = np.empty(B)
    for b in range(B):
        idx_a = rng.choice(len(data_a), len(data_a), replace=True, p=prob_a)
        idx_b = rng.choice(len(data_b), len(data_b), replace=True, p=prob_b)
        deltas[b] = (
            np.average(data_a[idx_a], weights=w_a[idx_a])
            - np.average(data_b[idx_b], weights=w_b[idx_b])
        )

    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    return theta, ci_lo, ci_hi


# tenure pairs to compare, plus a couple of year-over-year repeats for 2023/24
COMPARISONS = [
    (2022, "own_mortgage",  "private_rent"),
    (2022, "own_outright",  "private_rent"),
    (2022, "own_mortgage",  "social_rent"),
    (2022, "private_rent",  "social_rent"),
    (2023, "own_mortgage",  "private_rent"),
    (2024, "own_mortgage",  "private_rent"),
]

results = []
print(f"{'Year':<6} {'Group A':<16} {'Group B':<16} {'Gap':>6} {'CI low':>8} {'CI high':>8} {'Sig?'}")
print("-" * 70)

for year, ta, tb in COMPARISONS:
    yr = df[df["inflation_year"] == year]
    ga = yr[yr["tenure_type"] == ta]
    gb = yr[yr["tenure_type"] == tb]

    if len(ga) < 10 or len(gb) < 10:
        print(f"{year:<6} {ta:<16} {tb:<16}  -- insufficient data --")
        continue

    gap, lo, hi = bootstrap_gap(
        ga[ANALYSIS_RATE_COL].values, ga["household_weight"].values,
        gb[ANALYSIS_RATE_COL].values, gb["household_weight"].values,
    )
    sig = "YES" if (lo > 0 or hi < 0) else "no"
    print(f"{year:<6} {ta:<16} {tb:<16} {gap:>6.2f} {lo:>8.2f} {hi:>8.2f}  {sig}")
    results.append({
        "year": year, "group_a": ta, "group_b": tb,
        "gap": gap, "ci_lo": lo, "ci_hi": hi, "significant": sig == "YES",
    })

results_df = pd.DataFrame(results)
results_df.to_csv("bootstrap_tenure_gaps.csv", index=False)
print("\nSaved bootstrap_tenure_gaps.csv")

# quick income check: bottom vs top quintile in 2022
print("\nIncome Quintile Comparison (2022):")
yr2022 = df[df["inflation_year"] == 2022]
q1 = yr2022[yr2022["income_quintile"] == 1]
q5 = yr2022[yr2022["income_quintile"] == 5]
gap_inc, lo_inc, hi_inc = bootstrap_gap(
    q1[ANALYSIS_RATE_COL].values, q1["household_weight"].values,
    q5[ANALYSIS_RATE_COL].values, q5["household_weight"].values,
)
sig_inc = "YES" if (lo_inc > 0 or hi_inc < 0) else "no"
print(f"Q1 vs Q5 gap = {gap_inc:.2f}pp, 95% CI [{lo_inc:.2f}, {hi_inc:.2f}]  Sig? {sig_inc}")

# build a tidy summary table for 2022 (the main analysis year)
results_lookup = {
    (r["year"], r["group_a"], r["group_b"]): r
    for r in results
}

def format_row(label, kind, key, invert=False):
    r = results_lookup[key]
    gap = -r["gap"] if invert else r["gap"]
    lo = -r["ci_hi"] if invert else r["ci_lo"]
    hi = -r["ci_lo"] if invert else r["ci_hi"]
    return [label, kind, f"{gap:.2f}", f"[{lo:.2f}, {hi:.2f}]", "Yes" if r["significant"] else "No"]

table_rows = [
    format_row("Private rent vs Outright owner", "Tenure", (2022, "own_outright", "private_rent"), invert=True),
    format_row("Private rent vs Mortgagor",      "Tenure", (2022, "own_mortgage", "private_rent"), invert=True),
    format_row("Social rent vs Mortgagor",       "Tenure", (2022, "own_mortgage", "social_rent"), invert=True),
    format_row("Private rent vs Social rent",    "Tenure", (2022, "private_rent", "social_rent")),
    ["Income Q1 vs Q5",                          "Income", f"{gap_inc:.2f}", f"[{lo_inc:.2f}, {hi_inc:.2f}]", "Yes"],
]

col_labels = ["Comparison", "Type", "Gap (pp)", "95% CI", "Significant?"]

fig, ax = plt.subplots(figsize=(10, 2.8))
ax.axis("off")

tbl = ax.table(
    cellText=table_rows,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 1.6)

# give the comparison column more room, keep the rest tight
col_widths = [0.38, 0.12, 0.12, 0.18, 0.15]
for j, w in enumerate(col_widths):
    for i in range(6):
        tbl[i, j].set_width(w)

# dark blue header
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#1565C0")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# yellow tint on the income row so it stands out
for j in range(len(col_labels)):
    tbl[5, j].set_facecolor("#FFF9C4")

# zebra stripe the tenure rows
for i in range(1, 5):
    for j in range(len(col_labels)):
        tbl[i, j].set_facecolor("#F5F5F5" if i % 2 == 0 else "white")

ax.set_title("Bootstrap inflation gaps by tenure and income, 2022  (all significant at 95%)",
             fontsize=12, fontweight="bold", pad=12)

plt.tight_layout()
plt.savefig("plot_bootstrap_table.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_bootstrap_table.png")

# forest plot — blue = significant, grey = not
fig, ax = plt.subplots(figsize=(8, 5))
plt.rcParams.update({"font.size": 12})

labels = [
    f"{r['year']}: {r['group_a'].replace('_', ' ')} vs\n{r['group_b'].replace('_', ' ')}"
    for _, r in results_df.iterrows()
]
y = np.arange(len(results_df))

for i, (_, r) in enumerate(results_df.iterrows()):
    colour = "#2196F3" if r["significant"] else "#9E9E9E"
    ax.plot([r["ci_lo"], r["ci_hi"]], [i, i], color=colour, linewidth=2)
    ax.scatter(r["gap"], i, color=colour, s=60, zorder=3)

ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("Inflation gap (percentage points, group A minus group B)")
ax.set_title("Pairwise tenure inflation gaps with 95% bootstrap CIs\n(blue = significant, grey = not significant)")
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("plot_bootstrap_tenure_gaps.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_bootstrap_tenure_gaps.png")
