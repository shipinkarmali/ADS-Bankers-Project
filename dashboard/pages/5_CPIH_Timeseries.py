import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

sys.path.insert(0, f"{ROOT}/dashboard")

from visualise import OFFICIAL_PRICE_NAMES
from visualise import PRICE_MAP
from visualise import monthly
from visualise import cpih_time_series

import pandas as pd
import streamlit as st

price_cols = [c for c in OFFICIAL_PRICE_NAMES if c in monthly.columns]
options = {PRICE_MAP.get(c, c): c for c in price_cols}

selected_labels = st.sidebar.multiselect(
    "Select COICOP categories",
    options=list(options.keys()),
    default=list(options.keys())
)

selected_cols = [options[label] for label in selected_labels]

# Ensure datetime
monthly["date"] = pd.to_datetime(monthly["date"])

min_year = 2015
max_year = int(monthly["date"].dt.year.max())

start_year, end_year = st.sidebar.slider(
    "Select year range",
    min_value=min_year,
    max_value=max_year,
    value=(2015, 2026)
)

fig = cpih_time_series(monthly, selected_cols, start_year, end_year)

st.title("CPIH Time Series")

st.markdown("""
## CPIH Sub-Index Time Series with Key Events
            
### Sidebar Usage

Use the sidebar to add or remove COICOP categories from the figure or to adjust the year range.
            Removing Energy (elec/gas) will improve the visualisation of the disparity between other categories.
""")

st.pyplot(fig)

st.markdown("""
2015-2019 was a period of historically low inflation with headline CPIH remaining within the 0.4–2.6% range            and individual sub-indices moved broadly in step with limited dispersion around the headline rate.
            Crucially, When sub-indices move together, basket composition has little bearing on experienced inflation —
            regardless of how spending is allocated,
            all households face broadly the same price pressures.

From 2021, this pattern breaks.
            Energy prices surged to an index of approximately 235 by mid-2022 while actual rents remained below 115 —
            a gap of over 120 index points between two components that together form COICOP 04.
            Divergence in subindices prices makes the expenditure composition of a household
            much more consequential in how inflation impacts that household and
            makes average inflation estimates for a pooled mean household much less reliable.
            Private renters, who spend a much larger share of 'Housing and Utilities' cost on actual rent
            rather than energy, are largely shielded from the energy spike.
            Outright owners have the inverse relationship and were likely much more affected by the 2022 Energy Crisis.
            This divergence undermines the usefulness of a single headline inflation rate:
            during the 2022 energy crisis, the national CPIH reported one number,
            but the price shock experienced by an outright owner with high energy exposure
            bore little resemblance to that experienced by a private renter whose housing costs are predominantly rent.
""")