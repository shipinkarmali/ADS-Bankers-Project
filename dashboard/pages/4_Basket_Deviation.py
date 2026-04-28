import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

sys.path.insert(0, f"{ROOT}/dashboard")

from visualise import shares
from visualise import share_cols
from visualise import basket_deviation

fig = basket_deviation(shares, share_cols)

import streamlit as st

st.title("Basket Deviation")

st.markdown("""
## Tenure Basket Deviation from Pooled Mean
- Heatmap showing how each tenure group's spending deviates from the overall average.
- Bold values = deviation in percentage points, parentheses = raw share.
- This is the central EDA figure — it shows *where* tenure groups differ, not just *that* they differ.
""")

st.pyplot(fig)

st.markdown("""
The figure above shows where different tenure types allocate spending compared to the pooled mean.
            Owner occupied and renting households spend a greater and lesser share of total expenditure
            on transport than the pooled average respectively. This mirrors the UK property market structure
            where private renters are disproportionately located in urban areas compared to owner-occupiers,
            who are more concentrated in suburban locations. However,
            the biggest source of expenditure variation is by far housing and utilities (COICOP 04).
            While private and social renters allocate 40% and 32% of total expenditure
            resp. to this category (20.5 and 12.3 percentage points above the pooled mean resp.),
            outright owners and mortgage owners sit 8.8pp and 5.3pp below the mean respectively.
            This gap reflects the COICOP expenditure guidelines,
            under which renters' full rent is recorded as a consumption expense,
            whereas mortgage payments are classified as investments and excluded.
            Thus, owner-occupiers' Housing and Utilities spending therefore captures only energy,
            maintenence, water and council tax.
            While this is in line with the EU Harmonised Index of Consumer Prices methodology,
            we think this is a limitation as it suggests owner occupied househoulds with a mortgage only pay
            a 3.5pp greater share of spending on housing and utilities than owner occupied households without -
            understating the difference. 

This deviation in expenditure share on baskets directly impact
            how households of different tenure types experience inflation differently.
            For example, the £2 bus fare cap in 2022 (and later £3 cap)
            was intended to target a low-income demographic but our data indicates that
            the deflationary impact of reducing transport fees in general will impact
            social renters the least out of any tenure type.
            Social renters are widely understood to be the lowest income tenure type,
            suggesting a limitation in recent government policy.

The dominance of housing costs for renters produces a visible crowding-out effect in their baskets.
            Private renters fall below the pooled mean in ten of the remaining eleven divisions,
            with the largest deficits in Recreation (−4.6 pp), Food (−3.4 pp), and Transport (−3.2 pp).
            Social renters display a similar but less extreme compression,
            though they retain above-average Food (+2.3 pp) and Alcohol & Tobacco (+1.1 pp) shares —
            consistent with a lower-income demographic for whom necessities and habitual goods
            command a larger budget share even after housing costs.
""")