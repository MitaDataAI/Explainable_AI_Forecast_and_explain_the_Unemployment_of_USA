## Data Analysis vs Production Pipeline

### Exploratory Data Analysis (EDA)

The exploratory data analysis was conducted in Jupyter notebooks and is intentionally kept separate from the production codebase.

The EDA covers:
- Analysis of non-stationarity (ADF, KPSS tests, visual inspection)
- Justification of the chosen differencing orders and transformations
- Multi-dimensional correlation analysis (correlation matrices, collinearity diagnostics, exploratory visualizations)

These notebooks serve as:
- Scientific justification of modeling choices
- Documentation of the reasoning process
- Traceability for audit or review purposes

They are **not used nor executed in production**.

**Location:**
```text
notebooks/
 └── 01_eda.ipynb