import streamlit as st

st.set_page_config(
    page_title="Cost of Living Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Cost of Living and Inflation Dashboard")

st.markdown("""
#### Data preparation

This project relies on two data sets.
            The Living Costs and Food Survey (LCF) has demographic and expenditure data for each COICOP
            category at a household-level;
            it’s available via the UK Data Service and is used to construct our ‘spending baskets’
            for each housing tenure.
            The ONS MM23 supplies the CPIH price indices for each COICOP category,
            price indices for various subcategories within each COICOP category and an ‘All items CPIH Index’
            headline inflation rate.

#### LCF Microdata
The LCF is an expenditure survey for households conducted continuosly in the form of a biweekly expenditure diary.
            Each year, a fresh sample of around 5000 UK households per year.
""")

st.info("<- Select a page from the sidebar to begin.")