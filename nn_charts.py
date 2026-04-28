import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats


default_full_dir = "nn_household_outputs_full"
default_top6_dir = "nn_household_outputs_top6"

c_train = "#4472C4"
c_val   = "#ED1C24"
c_scatter = "#4472C4"
c_hist  = "#DCE6F2"
c_box   = "#ED1C24"
c_line  = "#222222"

font_size = 11
title_size = 13

plt.rcParams.update({
    "font.size": font_size,
    "axes.titlesize": title_size,
    "axes.labelsize": font_size,
    "xtick.labelsize": font_size - 1,
    "ytick.labelsize": font_size - 1,
    "legend.fontsize": font_size - 1,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "serif",
})


def load_outputs(outdir: str):
    history = pd.read_csv(os.path.join(outdir, "training_history.csv"))
    preds = pd.read_csv(os.path.join(outdir, "predictions_test_years.csv"))
    metrics = pd.read_csv(os.path.join(outdir, "test_metrics.csv"))
    return history, preds, metrics


# =========================================================
# CHART 1: LOSS CURVES
# =========================================================

def plot_loss_curves(history: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(history["epoch"], history["train_loss"],
            color=c_train, linewidth=1.5, label="Training loss")
    ax.plot(history["epoch"], history["val_loss"],
            color=c_val, linewidth=1.5, label="Validation loss")

    # Mark the early-stopping point (epoch with lowest val loss)
    best_idx = history["val_loss"].idxmin()
    best_epoch = history.loc[best_idx, "epoch"]
    best_val = history.loc[best_idx, "val_loss"]
    ax.axvline(best_epoch, color=c_line, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.scatter([best_epoch], [best_val], color=c_val, zorder=5, s=40,
               label=f"Best epoch ({int(best_epoch)})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted MSE")
    ax.set_title("Training and Validation Loss")
    ax.legend(frameon=False)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


# =========================================================
# CHART 2: PREDICTED vs ACTUAL
# =========================================================

def plot_pred_vs_actual(preds: pd.DataFrame, metrics: pd.DataFrame, save_path: str):
    y_true = preds["excess_household_inflation"]
    y_pred = preds["predicted_excess_inflation"]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    ax.scatter(y_true, y_pred, alpha=0.15, s=8, color=c_scatter,
               edgecolors="none", rasterized=True)

    # 45-degree reference line
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, color=c_line, linewidth=1, linestyle="--", alpha=0.6)

    r2 = metrics["test_r2"].iloc[0] if "test_r2" in metrics.columns else metrics["r2"].iloc[0]
    rmse_val = metrics["test_rmse"].iloc[0] if "test_rmse" in metrics.columns else metrics["rmse"].iloc[0]
    ax.text(0.05, 0.93,
            f"$R^2$ = {r2:.4f}\nRMSE = {rmse_val:.4f}",
            transform=ax.transAxes, fontsize=font_size,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))

    ax.set_xlabel("Actual excess household inflation")
    ax.set_ylabel("Predicted excess household inflation")
    ax.set_title("Predicted vs Actual")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


# =========================================================
# CHART 3: RESIDUAL DISTRIBUTION
# =========================================================

def plot_residual_distribution(preds: pd.DataFrame, save_path: str):
    residuals = preds["residual"]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(residuals, bins=80, density=True, color=c_hist,
            edgecolor="white", linewidth=0.3, alpha=0.85, label="Residuals")

    # Overlay a KDE
    x_kde = np.linspace(residuals.min(), residuals.max(), 300)
    kde = stats.gaussian_kde(residuals)
    ax.plot(x_kde, kde(x_kde), color=c_train, linewidth=1.5, label="KDE")

    # Zero line
    ax.axvline(0, color=c_line, linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_xlabel("Residual (actual − predicted)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Residuals on Test Set")
    ax.legend(frameon=False)

    # Annotate mean and std
    ax.text(0.95, 0.93,
            f"Mean = {residuals.mean():.4f}\nStd = {residuals.std():.4f}",
            transform=ax.transAxes, fontsize=font_size - 1,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


# =========================================================
# CHART 4: RESIDUALS BY TEST YEAR
# =========================================================

def plot_residuals_by_year(preds: pd.DataFrame, save_path: str):
    years = sorted(preds["year"].unique())

    fig, ax = plt.subplots(figsize=(5, 4))

    data = [preds.loc[preds["year"] == y, "residual"].values for y in years]

    bp = ax.boxplot(
        data,
        labels=[str(int(y)) for y in years],
        patch_artist=True,
        widths=0.5,
        showfliers=True,
        flierprops=dict(marker=".", markersize=2, alpha=0.3, color=c_line),
        medianprops=dict(color=c_line, linewidth=1.5),
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(c_box)
        patch.set_alpha(0.7)

    ax.axhline(0, color=c_line, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Test year")
    ax.set_ylabel("Residual (actual − predicted)")
    ax.set_title("Residuals by Test Year")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="NN diagnostic charts")
    parser.add_argument("--top6", action="store_true",
                        help="Read from the top-5 output directory")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Custom output directory to read from")
    args = parser.parse_args()

    if args.outdir:
        outdir = args.outdir
    elif args.top6:
        outdir = default_top6_dir
    else:
        outdir = default_full_dir

    if not os.path.isdir(outdir):
        raise FileNotFoundError(
            f"Output directory not found: {outdir}\n"
            f"Run the training script first, then run this."
        )

    history, preds, metrics = load_outputs(outdir)

    chart_dir = os.path.join(outdir, "charts")
    os.makedirs(chart_dir, exist_ok=True)

    plot_loss_curves(history, os.path.join(chart_dir, "loss_curves.png"))
    plot_pred_vs_actual(preds, metrics, os.path.join(chart_dir, "pred_vs_actual.png"))
    plot_residual_distribution(preds, os.path.join(chart_dir, "residual_distribution.png"))
    plot_residuals_by_year(preds, os.path.join(chart_dir, "residuals_by_year.png"))

    print(f"\nAll charts saved to: {chart_dir}/")


if __name__ == "__main__":
    main()
