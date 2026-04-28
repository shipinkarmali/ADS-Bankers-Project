import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

sys.path.insert(0, f"{ROOT}/dashboard")

from visualise import shares
from visualise import basket_evolution

import streamlit as st

min_year = int(shares["year"].min())
max_year = int(shares["year"].max())

start_year, end_year = st.sidebar.slider(
    "Select financial year range",
    min_value=min_year,
    max_value=max_year,
    value=(2015, 2023)
)

fig = basket_evolution(shares, start_year, end_year)

st.title("Tenure Basket Evolution")

st.markdown("""
### Sidebar Usage

Use the sidebar to adjust the year range
""")

st.pyplot(fig)

st.markdown("""
Fig. A tracks the share of expenditure on essentials (defined as food, energy and rent).
            There's a clear persistent structural divide that spans nine years:
            social and private renters allocated over 50% of total expenditure to essentials,
            compared with a third for owner-occupied households.
            Between 2019/20 and 2023/24,
            the essentials share rose by approximately 5–7pp across all four tenure groups —
            expenditure that was previously available for discretionary spending was diverted to
            absorb the rising food and energy costs.
            While this compression of disposable budgets was similar for all households in pp,
            it occurred on top of an already unequal baseline with private renters reaching 58% of total
            expenditure on essentials by 2023/24.
            They had much less discretionary spending to absorb the inflation spike compared to households
            that own outright who faced the same shock from a baseline of 25% of spending on essentials.
            Thus, it's likely that the real-world effects of this discretionary squeeze
            (caused by the cost of living crisis) affected renters disproportionately.
            Crucially, this disproportionate impact on households is something one headline CPIH inflation rate
            can't show.

Since the energy price sub-index rose the fastest,
            its effect on expenditure shares for tenure types was seperated into the RHS chart.
            Although social renters typically live in homes with higher median EPC scores (70)
            than owner occupied homes (64),
            their share of energy spending rises most significantly.
            However, this is likely because they have the lowest total expenditure.
            Private renters seem least affected by the energy crisis,
            with energy share rising only 1.5pp from 2019-23, own outright households' energy share grew triple
            this in the same period.
            This is consistent with a subset of private renters that had utilities included in their rent,
            shielding them (and their recorded energy expenditure) from the price increase.
""")