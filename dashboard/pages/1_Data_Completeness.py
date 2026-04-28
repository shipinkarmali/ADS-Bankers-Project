import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

sys.path.insert(0, f"{ROOT}/dashboard")

from visualise import shares
from visualise import share_cols
from visualise import data_completeness

fig = data_completeness(shares, share_cols)

import streamlit as st

st.title("Data Completeness")

st.markdown("""
## Data quality audit
- (a) How much data is genuinely missing per column?

- (b) How many households report 0 spending per COICOP division?
Note: This is not a data quality issue but reflects that LCF is a biweekly diary where
            households may have reasons for 0 spending in a category
            (most households don't pay for education or tobacco every 2 weeks).
""")

st.pyplot(fig)

st.markdown("""
The above figure audits data completeness and expenditure prevalence across the 12 COICOP divisions.
            Fig. A confirms that missingness is negligible:
            only Housing & Utilities exhibits any notable missing values (0.10%),
            with all other divisions at or below 0.01%.
            This reflects that LCF is designed as a structured diary where households record expenditure against
            pre-defined COICOP categories.

Fig. B distinguishes a lack of purchasing from data quality failures in certain categories.
            High zero-expenditure rates in Education (95.5%), Health (45.1%), and Clothing (39.4%) are expected:
            most households do not pay for private school or private healthcare at all,
            let alone within the two-week diary window.
            Clothing purchases were also infrequent within a two-week window.
            In contrast, essential categories such as Food (0.04%) and Housing & Utilities (0.2%)
            are nearly universal. Households reporting zero in these divisions likely submitted an incomplete diary.
            Consequently, these cases are removed during filtering: 296 for zero food,
            50 for zero housing, and 7 for negative total expenditure.
            In total, 350 households (0.76% of the sample) were removed.

Post-filtering, expenditure shares sum to exactly 1.0 for all 45,500 households.
""")
