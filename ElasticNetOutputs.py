import os
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ElasticNetFinal import features


outdir = "elasticnet_household_outputs"

theme = {
    "navy": "#2E4F8A",
    "black": "#222222",
    "light_grey": "#D9D9D9",
    "red": "#ED1C24",
    "blue": "#4472C4",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": theme["black"],
    "axes.labelcolor": theme["black"],
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.frameon": False,
    "grid.color": theme["light_grey"],
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "savefig.bbox": "tight",
})


def plot_coefficients(coef_df):
    # standardised features so coefficient magnitudes are directly comparable
    top = coef_df.head(15).copy()
    top["feature_pretty"] = top["feature"].apply(features)
    top = top.sort_values("abs_coefficient", ascending=True)

    # red = positive (pushes inflation up), blue = negative (pushes it down)
    colors = [theme["red"] if c > 0 else theme["blue"] for c in top["coefficient"]]

    plt.figure(figsize=(10.5, 6.8))
    plt.barh(
        top["feature_pretty"],
        top["coefficient"],
        color=colors,
        edgecolor=theme["navy"],
        alpha=0.9,
    )
    plt.axvline(0, color=theme["black"], linewidth=1)
    plt.xlabel("Standardised coefficient")
    plt.title("Elastic Net: top 15 coefficients (standardised features)")
    plt.grid(True, axis="x", alpha=0.35)

    legend_elements = [
        Patch(facecolor=theme["red"], edgecolor=theme["navy"], label="Increases excess inflation"),
        Patch(facecolor=theme["blue"], edgecolor=theme["navy"], label="Decreases excess inflation"),
    ]
    plt.legend(handles=legend_elements, loc="lower right")

    out = os.path.join(outdir, "fig1_coefficients.png")
    plt.savefig(out, dpi=300)
    plt.close()


def main():
    print("running elastic net outputs...")

    coef_df = pd.read_csv(os.path.join(outdir, "coefficients.csv"))
    plot_coefficients(coef_df)

    print(f"figure saved to {outdir}/")


if __name__ == "__main__":
    main()
