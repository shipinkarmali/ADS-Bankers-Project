import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, r2_score


csv_path = "cleaned/household_inflation_dataset.csv"
outdir    = "outputs/lr_outputs"
os.makedirs(outdir, exist_ok=True)

year_col       = "year"
weight_col     = "household_weight"
raw_target_col = "household_inflation_rate"
target_col     = "excess_household_inflation"

train_years = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
test_years  = [2022, 2023]

tenure_ref = "own_outright"
age_ref    = "50_to_64"

share_cols = [
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


def build_features(data, include_shares):
    # tenure dummies (ref = own_outright), age dummies (ref = 50_to_64), income as a continuous step
    td = pd.get_dummies(data["tenure_type"], prefix="tenure")
    td = td.drop(columns=[c for c in td.columns if tenure_ref in c], errors="ignore")
    ad = pd.get_dummies(data["hrp_age_band"], prefix="age")
    ad = ad.drop(columns=[c for c in ad.columns if age_ref in c], errors="ignore")
    inc = pd.to_numeric(data["income_quintile"], errors="coerce").rename("income_quintile")
    parts = [td.reset_index(drop=True), ad.reset_index(drop=True), inc.reset_index(drop=True)]
    if include_shares:
        parts.append(data[share_cols].reset_index(drop=True))
    return pd.concat(parts, axis=1)


def fit_wls(train_df, test_df, include_shares):
    x_tr = build_features(train_df, include_shares)
    x_te = build_features(test_df,  include_shares).reindex(columns=x_tr.columns, fill_value=0)
    y_tr = train_df[target_col].values
    y_te = test_df[target_col].values
    w_tr = train_df[weight_col].values

    ok_tr = x_tr.notna().all(axis=1)
    ok_te = x_te.notna().all(axis=1)
    x_tr, y_tr, w_tr = x_tr[ok_tr].reset_index(drop=True), y_tr[ok_tr], w_tr[ok_tr]
    x_te, y_te = x_te[ok_te].reset_index(drop=True), y_te[ok_te]

    xs_tr = sm.add_constant(x_tr.astype(float))
    mdl = sm.WLS(y_tr, xs_tr, weights=w_tr).fit()
    xs_te = sm.add_constant(x_te.astype(float)).reindex(columns=xs_tr.columns, fill_value=0)
    y_pred = mdl.predict(xs_te)

    coef = pd.DataFrame({
        "feature":     mdl.params.index,
        "coefficient": mdl.params.values,
        "std_err":     mdl.bse.values,
        "p_value":     mdl.pvalues.values,
        "ci_lower":    mdl.conf_int()[0].values,
        "ci_upper":    mdl.conf_int()[1].values,
        "significant": mdl.pvalues.values < 0.05,
    })
    return mdl, coef, y_pred, y_te


print("running linear regression pipeline...")

# load data and build the excess-inflation target (household inflation minus its year's weighted mean)
df = pd.read_csv(csv_path, encoding="utf-8-sig")
df.columns = df.columns.str.strip()
df = df.dropna(subset=[year_col, weight_col, raw_target_col]).copy()

tmp = df[[year_col, raw_target_col, weight_col]].copy()
tmp["wt"] = tmp[raw_target_col] * tmp[weight_col]
yw = tmp.groupby(year_col, as_index=False).agg(ws=("wt", "sum"), ww=(weight_col, "sum"))
yw["year_weighted_mean"] = yw["ws"] / yw["ww"]
df = df.merge(yw[[year_col, "year_weighted_mean"]], on=year_col, how="left")
df[target_col] = df[raw_target_col] - df["year_weighted_mean"]

train = df[df[year_col].isin(train_years)].copy().reset_index(drop=True)
test  = df[df[year_col].isin(test_years)].copy().reset_index(drop=True)

# Model 1 = demographics only; Model 2 = demographics + COICOP shares
mdl1, coef1, pred1, yte1 = fit_wls(train, test, include_shares=False)
mdl2, coef2, pred2, yte2 = fit_wls(train, test, include_shares=True)

m1 = {
    "model":    "Model 1 — demographics only (tenure + age + income)",
    "train_r2": round(mdl1.rsquared, 4),
    "test_r2":  round(r2_score(yte1, pred1), 4),
    "test_mae": round(mean_absolute_error(yte1, pred1), 4),
}
m2 = {
    "model":    "Model 2 — demographics + expenditure shares",
    "train_r2": round(mdl2.rsquared, 4),
    "test_r2":  round(r2_score(yte2, pred2), 4),
    "test_mae": round(mean_absolute_error(yte2, pred2), 4),
}
comparison = pd.DataFrame([m1, m2])

coef1.to_csv(f"{outdir}/model1_coefficients.csv", index=False)
coef2.to_csv(f"{outdir}/model2_coefficients.csv", index=False)
comparison.to_csv(f"{outdir}/model_comparison.csv", index=False)


# Figure 11 / 12 — predicted vs actual scatter for each model
for y_true, y_pred, m, label, fname in [
    (yte1, pred1, m1, "Model 1: Demographics Only",     "fig_model1_actual_vs_predicted.png"),
    (yte2, pred2, m2, "Model 2: Demographics + Shares", "fig_model2_actual_vs_predicted.png"),
]:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.05, s=5, color="#4472C4")
    lims = [min(y_true.min(), y_pred.min()) - 0.5, max(y_true.max(), y_pred.max()) + 0.5]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect fit")
    ax.set_xlabel("Actual excess inflation (pp)", fontsize=11)
    ax.set_ylabel("Predicted excess inflation (pp)", fontsize=11)
    ax.set_title(f"{label}\n(test years {test_years})", fontsize=11)
    ax.text(0.05, 0.95,
            f"Test R² = {m['test_r2']:.3f}\nMAE     = {m['test_mae']:.3f} pp",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/{fname}", dpi=150)
    plt.close()


# significance stars: *** p<0.001, ** p<0.01, * p<0.05, ns otherwise
def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# Figure 13 — Model 2 tenure coefficients vs own_outright with 95% CI
tp = coef2[coef2["feature"].str.startswith("tenure_")].copy()
tp["label"] = tp["feature"].str.replace("tenure_", "").str.replace("_", " ").str.title()
tp = tp.sort_values("coefficient", ascending=False).reset_index(drop=True)

bar_colors = ["#ED1C24" if v > 0 else "#4472C4" for v in tp["coefficient"]]
err_lo = tp["coefficient"] - tp["ci_lower"]
err_hi = tp["ci_upper"]   - tp["coefficient"]

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(tp["label"], tp["coefficient"], color=bar_colors, alpha=0.85, width=0.5)
ax.errorbar(tp["label"], tp["coefficient"], yerr=[err_lo, err_hi],
            fmt="none", color="black", capsize=6, linewidth=1.5)
for i, row in tp.iterrows():
    ax.text(i, row["ci_upper"] + 0.005, stars(row["p_value"]), ha="center", fontsize=12)

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Tenure Type", fontsize=11)
ax.set_ylabel(f"Excess inflation pp vs {tenure_ref}\n(all else equal)", fontsize=10)
ax.set_title(f"Model 2: Demographics + Expenditure Shares\n"
             f"(trained {min(train_years)}–{max(train_years)}; "
             f"ref = {tenure_ref}, age {age_ref}; error bars = 95% CI)", fontsize=10)
ax.text(0.98, 0.97,
        f"Train R² = {m2['train_r2']:.3f}\nTest R²  = {m2['test_r2']:.3f}",
        transform=ax.transAxes, fontsize=9, va="top", ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
plt.tight_layout()
plt.savefig(f"{outdir}/fig_model2_tenure_coefficients.png", dpi=150)
plt.close()


# summary digest of every number quoted in Section V-A and Table III
with open(f"{outdir}/summary.txt", "w", encoding="utf-8") as f:
    f.write("Household Linear Regression (WLS)\n\n")
    f.write(f"train years: {sorted(train[year_col].unique().tolist())}\n")
    f.write(f"test years : {sorted(test[year_col].unique().tolist())}\n\n")

    f.write("Model comparison:\n")
    f.write(comparison.to_string(index=False))
    f.write("\n\n")

    for label, coef in [
        ("Model 1 — demographics only", coef1),
        ("Model 2 — demographics + expenditure shares", coef2)]:
        f.write(f"{label}\n")
        f.write(f"tenure effects (vs {tenure_ref}):\n")
        
        for _, r in coef[coef["feature"].str.startswith("tenure_")].iterrows():
            sig = "ok" if r["significant"] else "ns"
            name = r["feature"].replace("tenure_", "").replace("_", " ")
            f.write(f"  {name:20s}  {r['coefficient']:+.4f} pp  p={r['p_value']:.4f}  {sig}\n")
        
        inc = coef[coef["feature"] == "income_quintile"].iloc[0]
        sig = "ok" if inc["significant"] else "ns"
        f.write(f"\nincome quintile : {inc['coefficient']:+.4f} pp per step  p={inc['p_value']:.4f}  {sig}\n\n")

print(f"all outputs saved to {outdir}/")
