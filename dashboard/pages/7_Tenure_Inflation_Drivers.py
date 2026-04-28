import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

sys.path.insert(0, f"{ROOT}/dashboard")

from visualise import PRICE_MAP
from visualise import coicop_contributions
from visualise import laspeyres_decomp

import streamlit as st


options = {
    PRICE_MAP.get(c, c): c
    for c in coicop_contributions["coicop_label"].unique()
}

selected_labels = st.sidebar.multiselect(
    "Select COICOP categories",
    options=list(options.keys()),
    default=list(options.keys())
)

selected_coicop = [options[label] for label in selected_labels]

fig = laspeyres_decomp(selected_coicop)

st.title("Tenure Inflation Drivers")

st.markdown("""
### Sidebar Usage
            
Use the sidebar to add or remove COICOP categories from the figure
""")

st.pyplot(fig)

st.markdown("""
The above figure shows the inflation each tenure group experienced during the 2022/23 FY energy crisis,
            which we computed using Laspeyres:
            2021/22 expenditure shares x price changes during FY 2022/2023 for each COICOP category.
            While the tenure types have structurally different expenditure baskets,
            this didn't lead to much variation in inflation in the low inflation environment of 2015-19.
            However, this pattern breaks during the energy crisis: unlike the previous years,
            where the inflation of each category tracked the composite CPIH inflation figure with little variation,
            2022-23 showed much greater variation between categories.
            Thus, the differences in how tenure types allocate their budgets had a significant effect during
            this period.

Private renters experienced the lowest inflation of all tenure groups (7.34pp) because their 'actual rent'
            budget was so high that less budget was allocated to other categories where prices surged.
            For all groups, food and energy were the two categories with the biggest contribution to inflation
            (in that order). However,
            these categories combined affected private renters the least with food (1.8pp) and energy (1.2pp)
            in particular contributing substantially less than all other tenure types.
            It seems that a pooled average CPIH systematically underestimates the impact of actual rent and
            overestimates the impact of all other COICOP categories for private tenants but this effect is
            more pronounced in food, energy and transport.
            For all other tenure types (including social renters),
            this effect is reversed by overestimating the impact of actual rent and underestimating
            the impact of other categories. Transport inflation diverged the most across tenure types,
            having a 44-100% greater impact on all homeowners (1.3-1.6pp) compared to all renters (0.8-0.9pp).

It seems that when inflation rates are low and consistent across subcategories,
            the headline CPIH figure is effective at estimating the impacts across all tenure types as
            they are affected similarly. However, when inflation rates increase and vary across subcategories,
            assessing the impact for each tenure type seperately provides useful insights that a
            headline CPIH figure could not capture alone.
""")