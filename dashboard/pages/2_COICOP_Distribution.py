import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

sys.path.insert(0, f"{ROOT}/dashboard")

from visualise import shares
from visualise import share_cols
from visualise import expenditure_shares

import streamlit as st

### SIDEBAR OPTIONS ###

show_skew = st.sidebar.checkbox("Show Skewness", value=True)
show_kurt = st.sidebar.checkbox("Show Kurtosis", value=True)

fig = expenditure_shares(shares, share_cols, show_skew, show_kurt)

st.title("COICOP Distribution")

st.markdown("""
## Distribution of COICOP Expenditure Shares
- (a) Boxplots showing the spread of each COICOP share across all households
- (b) Skewness and Kurtosis per share.
            
### Sidebar Usage

Use the sidebar to choose which statistical measures are shown in (B).
            It is reccommended to try turning Show Kurtosis off,
            to improve skewness clarity.        
""")

st.pyplot(fig)

st.markdown("""
(A) reveals considerable variation in spread of expenditure shares across COICOP divisions across all 45,500
            households. Food and Housing & Utilities show the highest median shares -
            consistent with being essential spending categories.
            In contrast, Education and Health are concentrated near 0% with a long right tail:
            some households allocate upwards of 10% of expenditure to Education.
            Public school and the NHS are widely used with a minority of households using private school or healthcare.
            Housing & Utilities show the widest IQR, highlighting the structural divide between renters and outright
            owners.

(B) further explains this asymmetry.
            Education, Health and Alcohol and Tobacco are most skewed (>2),
            confirming the long right tails in the boxplots.
            These categories show elevated excess kurtosis,
            indicating extreme values are more common than a normal distribution would predict.
            Food and Housing and Utilities are nearly symmetric,
            as they are universal expenditure categories for almost all households.

Households were only removed if they had zero spending on essential categories
            (categories where zero spending is implausible) or negative total spending,
            as this reflects an incomplete expenditure diary.
            296 with zero food expenditure, 50 with zero housing expenditure and 7 with negative total spending
            were removed.
            The heavily right-skewed distributions in categories like Education and Health are not outliers;
            they reflect genuine spending heterogeneity among households.
            Z-score filtering was considered but is unsuitable:
            with Education's mean near zero and 95.5% of households reporting no spending,
            any household paying tuition fees would be flagged as extreme despite representing a legitimate subpopulation.
            Winsorisation was also considered but is incompatible with compositional data:
            expenditure shares must sum to 1.0 per household, and capping one share breaks this constraint.
            Redistributing the clipped amount across the remaining categories would require an arbitrary allocation rule
            that distorts the Laspeyres weights.""")
