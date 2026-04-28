import os
import copy
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


dataset_path = r"cleaned\household_inflation_dataset.csv"
outdir = "nn_household_outputs"

seed = 42

year_col = "year"
weight_col = "household_weight"
raw_target_col = "household_inflation_rate"
target_col = "excess_household_inflation"

categorical_cols = [
    "tenure_type",
    "income_quintile",
    "hrp_age_band",
]

numeric_cols = [
    "household_size",
    "income_gross_weekly",
    "income_equivalised",
    "total_expenditure",
    "hrp_age",
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
    "share_04_actual_rent",
    "share_04_energy_other",
]

use_top6 = False

# Top 6 features from Random Forest permutation importance
top6_features = [
    "share_12_misc_goods_services",
    "share_03_clothing_footwear",
    "share_05_furnishings",
    "share_02_alcohol_tobacco",
    "share_09_recreation_culture",
    "share_11_restaurants_hotels",
]

outdir = outdir + ("_top6" if use_top6 else "_full")
os.makedirs(outdir, exist_ok=True)

train_years = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
test_years = [2022, 2023]

# hyperparameters
hidden_dims = (64, 32, 16) if use_top6 else (128, 64, 32)
dropout = 0.30 if use_top6 else 0.35
learning_rate = 1e-3
weight_decay = 1e-3
batch_size = 512
max_epochs = 200
early_stop_patience = 20
grad_clip = 1.0

val_fraction = 0.15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InflationMLP(nn.Module):
    """
    Feed-forward regressor for excess household inflation.
    Architecture: Input -> [Linear -> LayerNorm -> GELU -> Dropout] × N -> Linear(1)
    """
    def __init__(self, input_dim, hidden_dims=hidden_dims, dropout=dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def rmse(y_true, y_pred, weights=None):
    return float(np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=weights)))

def mae(y_true, y_pred, weights=None):
    return float(mean_absolute_error(y_true, y_pred, sample_weight=weights))

def r2(y_true, y_pred, weights=None):
    return float(r2_score(y_true, y_pred, sample_weight=weights))

def load_data(csv_path):
    print("Reading file from:", os.path.abspath(csv_path))
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    required_cols = (
        [year_col, weight_col, raw_target_col]
        + categorical_cols
        + numeric_cols
    )
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_cols].copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
    df[raw_target_col] = pd.to_numeric(df[raw_target_col], errors="coerce")
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan})
    df = df.dropna(subset=[year_col, weight_col, raw_target_col]).copy()

    # weighted yearly mean → excess target
    tmp = df[[year_col, raw_target_col, weight_col]].copy()
    tmp["weighted_target"] = tmp[raw_target_col] * tmp[weight_col]
    ywm = (tmp.groupby(year_col, as_index=False)
               .agg(ws=("weighted_target", "sum"), ww=(weight_col, "sum")))
    ywm["year_weighted_mean"] = ywm["ws"] / ywm["ww"]
    ywm = ywm[[year_col, "year_weighted_mean"]]
    df = df.merge(ywm, on=year_col, how="left")
    df[target_col] = df[raw_target_col] - df["year_weighted_mean"]

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna("unknown")

    df_raw = df.copy()
    df_model = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=np.uint8)
    return df_raw, df_model


def make_feature_cols(df_model):
    excluded = {target_col, raw_target_col, year_col, weight_col, "year_weighted_mean"}
    all_features = [c for c in df_model.columns if c not in excluded]
    if not use_top6:
        return all_features
    top6 = [c for c in top6_features if c in all_features]
    missing = [c for c in top6_features if c not in all_features]
    if missing:
        raise ValueError(f"top6 features missing from the dataset: {missing}")
    print(f"[top-6 mode] Using features: {top6}")
    return top6


def weighted_mse_loss(pred, target, weight):
    return ((pred - target) ** 2 * weight).sum() / (weight.sum() + 1e-12)


def train_model(X_fit, y_fit, w_fit, X_val, y_val, w_val, input_dim):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = InflationMLP(input_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-5,
    )

    X_fit_t = torch.tensor(X_fit, dtype=torch.float32, device=device)
    y_fit_t = torch.tensor(y_fit, dtype=torch.float32, device=device)
    w_fit_t = torch.tensor(w_fit, dtype=torch.float32, device=device)

    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
    w_val_t = torch.tensor(w_val, dtype=torch.float32, device=device)

    train_ds = TensorDataset(X_fit_t, y_fit_t, w_fit_t)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)

    best_val = float("inf")
    best_state = None
    patience = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        running_w = 0.0
        for X_b, y_b, w_b in train_loader:
            optimizer.zero_grad()
            pred = model(X_b)
            loss = weighted_mse_loss(pred, y_b, w_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running_loss += loss.item() * w_b.sum().item()
            running_w += w_b.sum().item()
        train_loss = running_loss / max(running_w, 1e-12)

        model.eval()
        with torch.no_grad():
            pred_val = model(X_val_t)
            val_loss = weighted_mse_loss(pred_val, y_val_t, w_val_t).item()

        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "lr": optimizer.param_groups[0]["lr"]})

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"  early stop at epoch {epoch}, best val MSE={best_val:.4f}")
                break

        if epoch == 1 or epoch % 20 == 0:
            print(f"  epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  best={best_val:.4f}")

    model.load_state_dict(best_state)
    return model, best_val, pd.DataFrame(history)

def run(df_raw, df_model):
    feature_cols = make_feature_cols(df_model)

    train_window = df_model[df_model[year_col].isin(train_years)].copy().reset_index(drop=True)
    test_df = df_model[df_model[year_col].isin(test_years)].copy().reset_index(drop=True)
    test_raw = df_raw[df_raw[year_col].isin(test_years)].copy().reset_index(drop=True)

    fit_idx, val_idx = train_test_split(
        np.arange(len(train_window)),
        test_size=val_fraction,
        random_state=seed,
        stratify=train_window[year_col].values,
    )
    fit_df = train_window.iloc[fit_idx].reset_index(drop=True)
    val_df = train_window.iloc[val_idx].reset_index(drop=True)

    numeric_to_scale = [c for c in feature_cols if c in numeric_cols]
    scaler = StandardScaler().fit(fit_df[numeric_to_scale].values)

    def scale_copy(d):
        out = d.copy()
        out[numeric_to_scale] = scaler.transform(d[numeric_to_scale].values)
        return out

    fit_df_s = scale_copy(fit_df)
    val_df_s = scale_copy(val_df)
    test_df_s = scale_copy(test_df)

    def to_arrays(d):
        return (d[feature_cols].values.astype(np.float32),
                d[target_col].values.astype(np.float32),
                d[weight_col].values.astype(np.float32))

    X_fit, y_fit, w_fit = to_arrays(fit_df_s)
    X_val, y_val, w_val = to_arrays(val_df_s)
    X_test, y_test, w_test = to_arrays(test_df_s)
    input_dim = X_fit.shape[1]

    print("=" * 60)
    print("HOUSEHOLD NEURAL NET — single model")
    print("=" * 60)
    print(f"Feature mode    : {'TOP-6' if use_top6 else 'FULL'}")
    print(f"Device          : {device}")
    print(f"Seed            : {seed}")
    print(f"Fit rows        : {len(fit_df)}   (stratified 85% of 2015–2021)")
    print(f"Val rows        : {len(val_df)}   (stratified 15% holdout)")
    print(f"Test rows       : {len(test_df)}   (2022–2023)")
    print(f"Input dim       : {input_dim}")
    print(f"Hidden dims     : {hidden_dims}")
    print(f"Dropout         : {dropout}")
    print(f"Weight decay    : {weight_decay}")
    print("Year distribution in fit/val/test:")
    print("  fit :", fit_df[year_col].value_counts().sort_index().to_dict())
    print("  val :", val_df[year_col].value_counts().sort_index().to_dict())
    print("  test:", test_df[year_col].value_counts().sort_index().to_dict())

    print("\n--- Training model ---")
    model, best_val, history = train_model(
        X_fit, y_fit, w_fit, X_val, y_val, w_val, input_dim,
    )

    model.eval()
    with torch.no_grad():
        preds_fit = model(torch.tensor(X_fit, dtype=torch.float32, device=device)).cpu().numpy()
        preds_val = model(torch.tensor(X_val, dtype=torch.float32, device=device)).cpu().numpy()
        preds_test = model(torch.tensor(X_test, dtype=torch.float32, device=device)).cpu().numpy()

    # ---- Metrics ----
    metrics = {
        "model": "NN_single",
        "seed": seed,
        "fit_n": len(fit_df),
        "val_n": len(val_df),
        "test_n": len(test_df),
        "train_r2": r2(y_fit, preds_fit, w_fit),
        "val_r2": r2(y_val, preds_val, w_val),
        "test_r2": r2(y_test, preds_test, w_test),
        "train_rmse": rmse(y_fit, preds_fit, w_fit),
        "val_rmse": rmse(y_val, preds_val, w_val),
        "test_rmse": rmse(y_test, preds_test, w_test),
        "train_mae": mae(y_fit, preds_fit, w_fit),
        "val_mae": mae(y_val, preds_val, w_val),
        "test_mae": mae(y_test, preds_test, w_test),
    }

    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)
    print(f"  Train  (fit) : R²={metrics['train_r2']:+.4f}  RMSE={metrics['train_rmse']:.4f}  MAE={metrics['train_mae']:.4f}")
    print(f"  Val    (15%) : R²={metrics['val_r2']:+.4f}  RMSE={metrics['val_rmse']:.4f}  MAE={metrics['val_mae']:.4f}")
    print(f"  Test  (22-23): R²={metrics['test_r2']:+.4f}  RMSE={metrics['test_rmse']:.4f}  MAE={metrics['test_mae']:.4f}")

    pd.DataFrame([metrics]).to_csv(os.path.join(outdir, "test_metrics.csv"), index=False)

    pred_df = test_raw[[year_col, weight_col, raw_target_col, target_col]].copy()
    pred_df["predicted_excess_inflation"] = preds_test
    pred_df["residual"] = pred_df[target_col] - pred_df["predicted_excess_inflation"]
    pred_df.to_csv(os.path.join(outdir, "predictions_test_years.csv"), index=False)

    history.to_csv(os.path.join(outdir, "training_history.csv"), index=False)
    fit_df_s.to_csv(os.path.join(outdir, "fit_processed.csv"), index=False)
    val_df_s.to_csv(os.path.join(outdir, "val_processed.csv"), index=False)
    test_df_s.to_csv(os.path.join(outdir, "test_processed.csv"), index=False)
    test_raw.to_csv(os.path.join(outdir, "test_raw.csv"), index=False)

    joblib.dump(feature_cols, os.path.join(outdir, "feature_cols.joblib"))
    joblib.dump(scaler, os.path.join(outdir, "scaler.joblib"))
    torch.save(model.state_dict(), os.path.join(outdir, "nn_model.pt"))

    return metrics, preds_fit, preds_val, preds_test, y_fit, y_val, y_test, w_fit, w_val, w_test, history


def main():
    df_raw, df_model = load_data(dataset_path)
    metrics, *_ = run(df_raw, df_model)
    print("\nSaved all outputs to:", outdir)


if __name__ == "__main__":
    main()
