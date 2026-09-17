# Urban Happiness: data, model and decision support system

Does the quality of the urban environment explain how people rate their lives?

This project merges eight public datasets into a country-year panel, fits a model
of life evaluation against six sustainability drivers, derives a composite **Urban
Happiness Index** from those drivers, and presents the result as an interactive
globe you can explore year by year.

The panel covers **122 countries** across **10 years** (2011 to 2021, with 2013
absent from the source data).

---

## Quick start

```bash
pip install -r requirements.txt
python dss.py
```

The dashboard opens at <http://127.0.0.1:8050>. Everything it needs is already in
`happiness_analysis_results.csv`, so you can run it without rebuilding anything.

To rebuild from source instead, see [Rebuilding the data](#rebuilding-the-data).

---

## Files

| File | What it is |
|---|---|
| `dss.py` | **The decision support system.** A 3-D globe with a year slider, driven by Dash and Plotly. |
| `dss_full_dashboard.py` | A fuller multi-chart dashboard over the same data: map, rankings, scatter with fit, trajectories, driver profiles and a scenario builder with adjustable weights. Run it the same way, on a different port: `python dss_full_dashboard.py --port 8060`. |
| `happiness_analysis.py` | Normalisation, OLS, XGBoost and PCA. Reads the merged panel, writes the results file. |
| `Data_merge.py` | Merges the eight source files into one panel. |
| `happiness_analysis_results.csv` | **What the DSS reads.** 920 rows, 23 columns. |
| `merged_whr_data_2011-2023_excl_2013.csv` | The merged panel before modelling. 1,822 rows, 168 countries, with gaps. |
| `requirements.txt` | Dependencies. `dss.py` itself needs only dash, plotly, pandas and numpy. |

Source data: `WHR.xlsx`, `GDP_PER_CAPITA.xls`, `HLE.csv`, `PM25_MEA.xls`,
`IHR.xls`, `DEGREE_OF _URBANIZATION.xlsx`, `SDG_Index.xlsx`.

---

## Using the DSS

- **Year slider**, pinned across the top. Every view below it follows the slider.
  2013 is not on the slider because it is not in the data.
- **Formula card**, showing how each index is built, with the PCA weights
  currently in force.
- **The globe**. Drag to spin it, scroll to zoom. Shading runs dark to bright,
  brighter meaning higher. Grey countries have no reading for the selected year.
  Two buttons above it switch the shading between Life Evaluation and the Urban
  Happiness Index.
- **Click any country** and its figures fill the panels either side. Indices and
  a ten-year trace on the left, drivers on the right with each one's weight in
  the index. Click the same country again to clear it.

The country you pick and the angle you spin to both survive a change of year, so
you can hold a view and step through time.

---

## The two indices

### Life Evaluation

The Cantril ladder from the World Happiness Report, taken as reported.
Respondents place their present life on a ladder from worst possible (0) to best
possible (10). This is a survey measure, not a formula, and it is the outcome
everything else is trying to explain.

### Urban Happiness Index (PCA weights)

```
UHI = 0.439·SDGₙ + 0.308·PM2.5ₙ + 0.197·Urbanisationₙ + 0.056·Homicideₙ
```

Each factor `ₙ` is min-max normalised to 1-10 across all countries and years. The
weighted sum is then rescaled to 1-10.

Weights are the absolute loadings of the first principal component of the four
factors, normalised to sum to 1. PC1 explains **51.7%** of the variance between
them.

| Factor | PC1 loading | Weight | Share of index |
|---|---:|---:|---:|
| SDG Index score | +0.764 | 0.439 | 43.9% |
| PM2.5 air pollution | −0.537 | 0.308 | 30.8% |
| Degree of urbanisation | −0.344 | 0.197 | 19.7% |
| Intentional homicide rate | −0.097 | 0.056 | 5.6% |

`dss.py` recomputes these weights from the data at startup rather than hard
coding them, so they stay correct if the underlying file changes. GDP per capita
and healthy life expectancy are measured and shown in the DSS but sit outside
this index, which is deliberately about the urban environment.

---

## Model results

OLS of life evaluation on all six normalised drivers, 920 observations:

**R² = 0.679, adjusted R² = 0.677**

| Driver | Coefficient | p-value |
|---|---:|---:|
| GDP per capita | +0.293 | 2.3 × 10⁻⁵⁸ |
| Intentional homicide rate | +0.252 | 6.6 × 10⁻²⁸ |
| Healthy life expectancy | +0.175 | 1.6 × 10⁻⁰⁹ |
| SDG Index score | +0.141 | 2.4 × 10⁻⁰⁷ |
| PM2.5 air pollution | −0.071 | 3.0 × 10⁻⁰⁴ |
| Degree of urbanisation | +0.028 | 0.070 |

Degree of urbanisation is the only driver that fails to clear p < 0.05.

On a held-out 20% test split:

| Model | RMSE | R² |
|---|---:|---:|
| OLS | 0.607 | 0.685 |
| XGBoost | 0.374 | 0.880 |

The gap between the two suggests real non-linearity in the relationship, which
the linear index cannot capture.

---

## Known issues

These are properties of the analysis as it stands, not bugs in the dashboard.
They are listed here so anyone reading the output knows what they are looking at.

**1. PM2.5 and homicide are not inverted before normalisation.**
`happiness_analysis.py` min-max scales every driver to 1-10 without flipping the
two where a lower reading is the better one. Both therefore enter the composite
indices with a positive sign, so a country with dirtier air scores *higher* on
the Urban Happiness Index. PM2.5 correlates −0.40 with life evaluation while
adding positively to the index built to track it.

The same shows up in the OLS table above: the homicide coefficient is **+0.252**,
which reads as "more homicide, higher life evaluation". It is an artefact of the
missing inversion, not a finding.

The DSS states the direction on every axis, colourbar and label, and the formula
card spells out the sign problem. `dss_full_dashboard.py` goes further: its
scenario builder re-orients both drivers before weighting, so a higher weight
there always means you want cleaner air and safer streets.

**2. `sustainable_happiness_index` duplicates `pca_happiness_index`.**
The two columns are byte-identical in this run. `happiness_analysis.py` assigns
the PCA index to both names without changing it. Neither appears in `dss.py`.

**3. Coverage falls away from 168 countries to 122.**
The merged panel holds 1,822 rows, but only 920 have every driver present, and
`happiness_analysis.py` drops the rest. The homicide rate is the worst offender
at 695 missing values, then healthy life expectancy at 441.

**4. The panel stops at 2021, not 2023.**
The merge reaches 2023, but no 2022 or 2023 row has a complete set of drivers, so
complete-case filtering removes both years entirely.

**5. Weights come from an unrotated PC1 explaining half the variance.**
At 51.7%, the first component leaves a lot on the table, and the weights inherit
whatever correlation structure the four factors happen to have.

---

## Rebuilding the data

Two stages, in order. Both write to the current directory.

```bash
python Data_merge.py        # 8 source files  ->  merged_whr_data_2011-2023_excl_2013.csv
python happiness_analysis.py # merged panel   ->  happiness_analysis_results.csv
```

`happiness_analysis.py` prints the full OLS summary, the p-value and PCA weights,
and the OLS against XGBoost comparison as it runs. It is deterministic: rerunning
it reproduces `happiness_analysis_results.csv` exactly.

**Stage 1 needs two Excel engines that are easy to miss.** `openpyxl` reads the
`.xlsx` sources and `xlrd` reads the three legacy `.xls` files. Without `xlrd`,
`Data_merge.py` fails at the GDP step with `Missing optional dependency 'xlrd'`.
Both are in `requirements.txt`.

On Windows, redirecting the merge output to a file also fails, because the script
prints a `✓` character that the console's default cp1252 encoding cannot
represent. Set the encoding first if you want a log:

```bash
PYTHONIOENCODING=utf-8 python Data_merge.py > merge.log 2>&1
```

To point the analysis at a different panel:

```bash
python happiness_analysis.py path/to/other_panel.csv
```

---

## Command line options

Both dashboards take the same flags:

```bash
python dss.py --data path/to/results.csv   # read a different results file
python dss.py --port 8060                  # serve on another port
python dss.py --host 0.0.0.0               # bind for access from another machine
python dss.py --no-browser                 # do not open a browser on start
python dss.py --debug                      # Dash debug mode with hot reload
```

---

## Data sources

| Driver | Source |
|---|---|
| Life evaluation | World Happiness Report, three-year average ladder score |
| GDP per capita | World Bank, purchasing power parity |
| Healthy life expectancy | World Health Organization, at birth |
| PM2.5 air pollution | Population-weighted mean concentration |
| Intentional homicide rate | UNODC, per 100,000, used as a proxy for crime |
| Degree of urbanisation | Share of population in urban settlements |
| SDG Index score | Sustainable Development Report composite |

---

## Requirements

Python 3.9 or later.

```
pandas  numpy  scikit-learn  statsmodels  xgboost  openpyxl  xlrd  dash  plotly
```

The dashboards need only pandas, numpy, dash and plotly, so if you are just
running `dss.py` against the committed CSV, that is all you have to install.

`openpyxl` and `xlrd` are needed only to rebuild from the Excel sources.
XGBoost is optional: `happiness_analysis.py` skips the model comparison and
reports OLS alone if it is not installed.
