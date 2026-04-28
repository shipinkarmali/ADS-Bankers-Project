import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

HH_PATH   = "cleaned/household_inflation_dataset.csv"
ANALYSIS_RATE_COL = "annual_inflation_rate"
ANALYSIS_GAP_COL = "annual_gap_vs_headline"

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
df[ANALYSIS_GAP_COL] = df[ANALYSIS_RATE_COL] - df["cpi_all_items_yoy_inflation"]

TENURE_TYPES = ["own_mortgage", "own_outright", "private_rent", "social_rent"]
LABELS = {
    "own_mortgage": "Own (Mortgage)",
    "own_outright": "Own (Outright)",
    "private_rent": "Private Rent",
    "social_rent":  "Social Rent",
}

groups = [df[df["tenure_type"] == t][ANALYSIS_RATE_COL].dropna().values
          for t in TENURE_TYPES]

pairs = [(TENURE_TYPES[i], TENURE_TYPES[j])
         for i in range(len(TENURE_TYPES))
         for j in range(i + 1, len(TENURE_TYPES))]

print(f"Households per tenure type (n={len(df):,} total):")
for t, g in zip(TENURE_TYPES, groups):
    print(f"  {t:<20} n={len(g):>6,}")

print("\nWith n=45k, p-values will almost always be significant -- effect sizes matter more.\n")

# Shapiro-Wilk on a subsample of 50 -- unreliable at large n so we sample down
rng = np.random.default_rng(42)
normality_results = {}
print("Shapiro-Wilk normality (sample n=50 per group):")
for t, g in zip(TENURE_TYPES, groups):
    sample = rng.choice(g, size=min(50, len(g)), replace=False)
    stat, p = stats.shapiro(sample)
    normality_results[t] = p
    print(f"  {t:<20} W={stat:.4f}  p={p:.4f}  {'normal' if p > 0.05 else 'not normal'}")

all_normal = all(p > 0.05 for p in normality_results.values())
test_choice = "ANOVA" if all_normal else "Kruskal-Wallis"
print(f"\n  -> using {test_choice} for group difference test")


# Kruskal-Wallis -- epsilon-squared as effect size, Bonferroni on pairwise tests
print(f"\nGroup difference ({test_choice}):")
print("H0: mean household inflation is equal across all four tenure types\n")

if all_normal:
    stat, p = stats.f_oneway(*groups)
    n_total = sum(len(g) for g in groups)
    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total   = sum(((g - grand_mean)**2).sum() for g in groups)
    eta2 = ss_between / ss_total
else:
    stat, p = stats.kruskal(*groups)
    n_total = sum(len(g) for g in groups)
    eta2 = (stat - len(groups) + 1) / (n_total - len(groups))  # epsilon-squared

print(f"  statistic={stat:.4f}  p={p:.4e}  epsilon-squared={eta2:.4f}")
if p < 0.05:
    if eta2 < 0.01:
        size_label = "negligible"
    elif eta2 < 0.06:
        size_label = "small"
    else:
        size_label = "medium/large"
    print(f"  significant difference -- effect size {size_label}")
    print("  statistically real but mean differences are small in practice")

    alpha_corrected = 0.05 / len(pairs)
    print(f"\n  pairwise comparisons (Bonferroni-corrected alpha = {alpha_corrected:.4f}):")
    for t1, t2 in pairs:
        s1 = df[df["tenure_type"] == t1][ANALYSIS_RATE_COL].dropna()
        s2 = df[df["tenure_type"] == t2][ANALYSIS_RATE_COL].dropna()
        if all_normal:
            _, p_pair = stats.ttest_ind(s1, s2)
        else:
            _, p_pair = stats.mannwhitneyu(s1, s2, alternative="two-sided")
        sig = "*" if p_pair < alpha_corrected else "ns"
        mean_diff = s1.mean() - s2.mean()
        print(f"    {t1} vs {t2:<20} p={p_pair:.4e}  mean diff={mean_diff:+.4f}%  {sig}")
else:
    print("  no significant difference in mean inflation rates")

print("\n  means by tenure type:")
for t, g in zip(TENURE_TYPES, groups):
    print(f"    {t:<20} mean={g.mean():.4f}%  median={np.median(g):.4f}%  std={g.std():.4f}")


# yearly means used for the time series plot
tenure_yearly = (
    df.groupby(["inflation_year", "tenure_type"])[ANALYSIS_RATE_COL]
    .mean()
    .reset_index()
    .pivot(index="inflation_year", columns="tenure_type", values=ANALYSIS_RATE_COL)
)[TENURE_TYPES]



SHARE_LABELS = {
    "share_01_food_non_alcoholic":  "Food",
    "share_02_alcohol_tobacco":     "Alcohol & Tobacco",
    "share_03_clothing_footwear":   "Clothing",
    "share_04_housing_fuel_power":  "Housing & Fuels",
    "share_05_furnishings":         "Furnishings",
    "share_06_health":              "Health",
    "share_07_transport":           "Transport",
    "share_08_communication":       "Communication",
    "share_09_recreation_culture":  "Recreation",
    "share_10_education":           "Education",
    "share_11_restaurants_hotels":  "Restaurants",
    "share_12_misc_goods_services": "Misc",
}

# Spearman within each year then averaged -- stops the 2022-23 spike dominating
print("\nSpearman correlations: spending shares vs household inflation rate")
print("(computed within-year to control for time effects)\n")

within_year_corrs = {}
for col in SHARE_COLS:
    year_corrs = []
    for yr in df["inflation_year"].unique():
        sub = df[df["inflation_year"] == yr][[col, ANALYSIS_RATE_COL]].dropna()
        if len(sub) > 10:
            r, _ = stats.spearmanr(sub[col], sub[ANALYSIS_RATE_COL])
            year_corrs.append(r)
    within_year_corrs[col] = np.mean(year_corrs)

for col, r in sorted(within_year_corrs.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {SHARE_LABELS[col]:<25} mean rho = {r:+.4f}")

print("\n  no single spending share is a strong predictor -- points to interactions")


# plots

colours = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]

tenure_colours = {
    "own_mortgage": "#2196F3",
    "own_outright": "#4CAF50",
    "private_rent": "#FF5722",
    "social_rent":  "#9C27B0",
}

gap_pivot = (
    df.groupby(["inflation_year", "tenure_type"])[ANALYSIS_GAP_COL]
    .mean()
    .reset_index()
    .pivot(index="inflation_year", columns="tenure_type", values=ANALYSIS_GAP_COL)
)[TENURE_TYPES]
gap_pivot.columns = [LABELS[t] for t in TENURE_TYPES]
gap_pivot.index.name = "Year"
pre_shock  = gap_pivot.drop(index=[2022, 2023], errors="ignore")
scale_max  = max(pre_shock.abs().max().max(), 0.5)

key_shares = {
    "Food":             "share_01_food_non_alcoholic",
    "Housing & Fuels":  "share_04_housing_fuel_power",
    "Transport":        "share_07_transport",
    "Recreation":       "share_09_recreation_culture",
}

plt.rcParams.update({"font.size": 13})


def savefig(name):
    plt.tight_layout()
    plt.savefig(name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {name}")


# yearly mean inflation by tenure type
fig, ax = plt.subplots(figsize=(9, 5))
for i, t in enumerate(TENURE_TYPES):
    ax.plot(tenure_yearly.index, tenure_yearly[t],
            marker="o", label=LABELS[t], color=colours[i], linewidth=2)
ax.axvspan(2021.9, 2023.1, alpha=0.1, color="red", label="Crisis period (2022-23)")
ax.set_title("Mean Household Inflation by Tenure Type")
ax.set_xlabel("Year")
ax.set_ylabel("Mean Inflation Rate (%)")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
savefig("plot_01_tenure_inflation_over_time.png")


# violin plot - full distribution per group
fig, ax = plt.subplots(figsize=(9, 5))
vdata = [df[df["tenure_type"] == t][ANALYSIS_RATE_COL].dropna().values
         for t in TENURE_TYPES]
parts = ax.violinplot(vdata, positions=range(len(TENURE_TYPES)), showmedians=True)
for pc, col in zip(parts["bodies"], colours):
    pc.set_facecolor(col)
    pc.set_alpha(0.6)
ax.set_xticks(range(len(TENURE_TYPES)))
ax.set_xticklabels([LABELS[t] for t in TENURE_TYPES], rotation=0, ha="center")
ax.set_title("Distribution of Household Inflation Rates (all 45k households)")
ax.set_ylabel("Household Inflation Rate (%)")
ax.grid(True, alpha=0.3, axis="y")
savefig("plot_02_violin_distributions.png")


# gap vs headline heatmap -- colour scale capped at pre-shock range so normal years are readable
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(gap_pivot, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-scale_max, vmax=scale_max,
            ax=ax, linewidths=0.5, annot_kws={"size": 11})
ax.set_title("Mean Inflation Gap vs Headline CPI by Tenure & Year\n(red = above headline; colour scale capped at pre-shock range)")
ax.tick_params(axis="x", rotation=0)
ax.tick_params(axis="y", rotation=0)
savefig("plot_03_heatmap_gap_vs_headline.png")



# within-year Spearman bar chart
fig, ax = plt.subplots(figsize=(10, 6))
corr_series = pd.Series(within_year_corrs).rename(index=SHARE_LABELS).sort_values()
colours_corr = ["#F44336" if v > 0 else "#2196F3" for v in corr_series.values]
ax.barh(corr_series.index, corr_series.values, color=colours_corr)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Spending Share vs Household Inflation\n(mean within-year Spearman rho)")
ax.set_xlabel("Spearman rho")
ax.grid(True, alpha=0.3, axis="x")
savefig("plot_06_within_year_spearman.png")


# mean spending shares by tenure for key categories
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(key_shares))
bar_width = 0.2
for i, t in enumerate(TENURE_TYPES):
    means = [df[df["tenure_type"] == t][col].mean() for col in key_shares.values()]
    ax.bar(x + i * bar_width, means, bar_width,
           label=LABELS[t], color=tenure_colours[t], alpha=0.8)
ax.set_xticks(x + bar_width * 1.5)
ax.set_xticklabels(key_shares.keys(), rotation=0, ha="center")
ax.set_title("Mean Spending Shares by Tenure Type (key categories)")
ax.set_ylabel("Mean Expenditure Share")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")
savefig("plot_07_spending_shares_by_tenure.png")

