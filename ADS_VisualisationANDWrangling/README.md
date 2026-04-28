# ADS Exploration Showcase
Note: Raw files were too big to include in GitHub (without paying for LFS). You can download all the raw stata LCF files from the UK Data Service 2015/16 - 2023/2024 (they will need to be downloaded individually). Adding them to the public repo would also be against the EUL that we agreed to when downloading them from the UK Data Service. THERE IS NO NEED TO DO THIS because the cleaned output files are already here. This is only neccessary if you want to re-create the cleaned output files for reproducibility.

### Dependencies
- Python 3.11+
- Dependencies are listed in requirements.txt
- Use our environment (environment setup)

### How you should set up the environment

- run these all from /ADS_VisualisationANDWrangling

```
python -m venv .venv
source .venv/bin/activate  
pip install -U pip
pip install -r requirements.txt
```

### Reload the databases (they are already here produced here in the data/outputs).
This creates the main output from raw LCF files and MM23_cleaned (a manually wrangled excel file that is used so we can access MM23 financial years rather than calendar years to match the temporal units of LCF).

```
python src/wrangle_lcf.py
python src/compute_group_inflation.py
```

- the new datasets will be created in data/output (they are already there)
- the next step is to just run all the cells in the .ipynb files in order in visualisations/
- There are two, one for the report diagrams (as they have to be viewable on one column of two column IEEE format) and one that was used for the website (hosted on streamlit). They are very similar except a few formatting differences.

### Viewing Visualisations EASILY
If you want to view the .png visualisations, you can just see them immediately in `visualisation_diagrams/` (at the repo root)
