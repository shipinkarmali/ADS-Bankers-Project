import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

sys.path.insert(0, f"{ROOT}/dashboard")

from visualise import shares
from visualise import summary_statistics

df = summary_statistics(shares)

import streamlit as st

st.title("Tenure statistics")

st.markdown("""
## Summary Statistics by Tenure

Weighted means of key expenditure shares and income, disaggregated by tenure group.
Shows the structural differences in household budgets that drive differential inflation.  
""")

st.dataframe(df)

st.markdown("""
Weighted mean expenditure shares (%) and gross weekly income by tenure group,
            pooled across 2015/16--2023/24. Means are weighted using LCF survey grossing weights.
            N denotes unweighted household-year observations.
            Mortgagor shares are computed over COICOP consumption expenditure,
            which excludes mortgage payments; their non-housing shares are therefore inflated relative to renters,
            whose housing costs enter the denominator directly.
""")