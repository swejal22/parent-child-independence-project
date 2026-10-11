# Parent–Child Bonds and Independence

This repository supports the CSCI 5612 website project examining communication,
emotional support, living arrangements, and financial independence among young
adults in the United States.

## Live website

[Parent–Child Bonds and Independence](https://parent-child-independence.streamlit.app/)

## Module 1: data preparation and EDA

- `01_Load_and_Filter_Pew_Data.ipynb` prepares the Pew young-adult sample.
- `02_Census_API_Fetch.ipynb` collects and cleans U.S. Census ACS data.
- `pew_young_adults_eda_labeled.csv` contains readable survey categories.
- `pew_young_adults_clean_numeric.csv` contains coded numeric survey data.

## Module 2: clustering and PCA

- `module2_analysis.py` creates the prepared model input and reproducible outputs.
- `03_Clustering_Analysis.ipynb` compares K-Means solutions and performs
  average-linkage hierarchical clustering with cosine distance.
- `04_PCA_Analysis.ipynb` performs PCA and interprets explained variance and
  component loadings.
- `module2_unlabeled_numeric_data.csv` is the 1,495-row, 12-feature unlabeled
  numeric dataset used by both methods.
- `module2_pages.py` contains the complete Streamlit Clustering and PCA pages.

The Streamlit application is launched from `streamlit_app.py`.
