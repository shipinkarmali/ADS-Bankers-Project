# ADS-Bankers-Project


## Run order

### 1. Data prep (run once, in order)
```bash
python mm23processing.py # Data/mm23.csv -> cleaned/mm23_key_categories_yoy.csv
python mm23toyear.py # -> cleaned/mm23_yearly_for_merge.csv 
python lcfmm23merge.py # + cleaned/lcf_expenditure_shares.csv -> cleaned/household_inflation_dataset.csv 

```

### 2. Statistical tests (independent, any order)
```bash
# tenure-group tests, ADF, Levene, shock-vs-non-shock, within-year Spearman + 9 plots
python statistical_tests.py
# weighted bootstrap of tenure/income inflation gaps -> bootstrap_tenure_gaps.csv + 2 plots
python bootstrap_tenure_gaps.py
```

### 3. Models (independent, any order)
```bash
python baselines.py                # outputs/baseline_outputs/
python linear_regression_final.py  # outputs/lr_outputs/

python ElasticNetFinal.py          # elasticnet_household_outputs/
python ElasticNetOutputs.py        # charts for elastic net

python RandomForestFinal.py        # rf_household_outputs/
python Randomforestoutputs.py      # charts + importances regarding Random Forest

python nn.py                       # nn_household_outputs_full/  (set use_top6=True for top 6)
python nn_charts.py                # add --top6 to read top-6 dir
```

## Dashboard

This project includes an interactive dashboard built using Streamlit and Matplotlib.

### Installation

To run the dashboard, install the required Python libraries:

```bash
pip install streamlit matplotlib pandas
```

or

```bash
conda install streamlit matplotlib pandas
```

### Running the Dashboard

Once the dependencies are installed, run the following command inside the `dashboard` directory:

```bash
streamlit run Dashboard.py
```

The dashboard will automatically open in your default browser.

# ADS-Bankers

## Files kept for marking

The repository keeps the scripts, required input CSVs, 
compact result CSVs, summary text files, and figures. 
Large generated row-level outputs and fitted model binaries are
excluded because they are reproducible from the scripts below.

Required input files:
```text
Data/mm23.csv
cleaned/lcf_expenditure_shares.csv
```
