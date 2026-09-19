#!/usr/bin/env python3
"""
Urban Happiness DSS - globe
===========================

A simple decision support view over ``happiness_analysis_results.csv``:

    * a time slider across the top of the window
    * the index formulae beneath it, with the PCA weights spelled out
    * a rotatable 3-D globe in the centre
    * click any country and its figures fill the panels either side

Drag the globe to spin it. The rotation, and the country you picked, both
survive a change of year.

Run:
    python dss.py
    python dss.py --data path/to/results.csv --port 8060 --no-browser

Then open http://127.0.0.1:8050

Requires: dash, plotly, pandas, numpy
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path
from threading import Timer

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

DEFAULT_CSV = Path(__file__).with_name("happiness_analysis_results.csv")
FONT_STACK = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

# ---------------------------------------------------------------------------
# Palette: black ground, yellow ink
# ---------------------------------------------------------------------------

BG = "#000000"
PANEL = "#0d0d0b"
BORDER = "rgba(255,216,77,0.20)"
RULE = "rgba(255,216,77,0.10)"

YELLOW = "#ffd84d"        # primary text
YELLOW_HI = "#ffe98f"     # values and emphasis
YELLOW_MID = "#c79a33"    # labels
YELLOW_LOW = "#8a6c24"    # the quietest chrome

OCEAN = "#0a0a09"
LAND = "#262523"          # a country with no reading: neutral, never amber
GRATICULE = "#1c1b17"
SELECTED = "#ffffff"      # the outline on the country you picked

# Single hue, dark to bright, so brighter always means more.
RAMP = ["#4a3507", "#63470a", "#7c590d", "#956b10", "#ae7d13",
        "#c79016", "#dba327", "#e9b845", "#f4cd68", "#fbdf92", "#ffecb8"]
COLOURSCALE = [[i / (len(RAMP) - 1), c] for i, c in enumerate(RAMP)]

# ---------------------------------------------------------------------------
# What the panels show
# ---------------------------------------------------------------------------
# ``better`` sets which end of the scale is rank 1. PM2.5 and the homicide
# rate are the two where a lower reading is the better one.

INDICES = [
    ("life_evaluation", "Life Evaluation", "Cantril ladder, 0-10", ",.2f", "up"),
    ("urban_happiness_index_pca", "Urban Happiness Index", "PCA weights, 1-10", ",.2f", "up"),
]

# The four normalised factors the Urban Happiness Index is built from, in the
# order the PCA weights rank them.
COMPONENTS = [
    ("sdg_index_score", "SDG Index score", "0-100", ",.1f", "up", "sdg_index_score_n"),
    ("pm2_5", "PM2.5 air pollution", "ug/m3, lower is better", ",.1f", "down", "pm2_5_n"),
    ("degree_of_urbanization", "Degree of urbanisation", "% urban", ",.1f", "up",
     "degree_of_urbanization_n"),
    ("intentional_homicide_rate", "Intentional homicide rate", "per 100k, lower is better",
     ",.2f", "down", "intentional_homicide_rate_n"),
]

# Measured, and strong predictors of the ladder, but outside the urban index.
CONTEXT = [
    ("gdp_per_capita", "GDP per capita", "int. $ PPP", ",.0f", "up"),
    ("healthy_life_expectancy", "Healthy life expectancy", "years", ",.1f", "up"),
]

PCA_FACTORS = [c[5] for c in COMPONENTS]
SHORT = {
    "sdg_index_score": "SDG",
    "pm2_5": "PM2.5",
    "degree_of_urbanization": "Urbanisation",
    "intentional_homicide_rate": "Homicide",
}

GLOBE_OPTIONS = [
    ("life_evaluation", "Life Evaluation"),
    ("urban_happiness_index_pca", "Urban Happiness Index"),
]
RANKED = [f[:5] for f in INDICES] + [c[:5] for c in COMPONENTS] + list(CONTEXT)

# ---------------------------------------------------------------------------
# Country reference: ISO 3166 alpha-3 so the globe can place each country
# ---------------------------------------------------------------------------

COUNTRY_REF = {
    # Africa
    "Algeria": ("DZA", "Africa"), "Angola": ("AGO", "Africa"),
    "Benin": ("BEN", "Africa"), "Botswana": ("BWA", "Africa"),
    "Burkina Faso": ("BFA", "Africa"), "Burundi": ("BDI", "Africa"),
    "Cameroon": ("CMR", "Africa"), "Eswatini": ("SWZ", "Africa"),
    "Ethiopia": ("ETH", "Africa"), "Ghana": ("GHA", "Africa"),
    "Kenya": ("KEN", "Africa"), "Liberia": ("LBR", "Africa"),
    "Malawi": ("MWI", "Africa"), "Mauritania": ("MRT", "Africa"),
    "Mauritius": ("MUS", "Africa"), "Morocco": ("MAR", "Africa"),
    "Mozambique": ("MOZ", "Africa"), "Namibia": ("NAM", "Africa"),
    "Niger": ("NER", "Africa"), "Nigeria": ("NGA", "Africa"),
    "Rwanda": ("RWA", "Africa"), "Senegal": ("SEN", "Africa"),
    "Sierra Leone": ("SLE", "Africa"), "Tunisia": ("TUN", "Africa"),
    "Uganda": ("UGA", "Africa"), "Zambia": ("ZMB", "Africa"),
    "Zimbabwe": ("ZWE", "Africa"),
    # Asia
    "Afghanistan": ("AFG", "Asia"), "Armenia": ("ARM", "Asia"),
    "Azerbaijan": ("AZE", "Asia"), "Bahrain": ("BHR", "Asia"),
    "Bangladesh": ("BGD", "Asia"), "Bhutan": ("BTN", "Asia"),
    "Cambodia": ("KHM", "Asia"), "China": ("CHN", "Asia"),
    "Georgia": ("GEO", "Asia"), "India": ("IND", "Asia"),
    "Iraq": ("IRQ", "Asia"), "Israel": ("ISR", "Asia"),
    "Japan": ("JPN", "Asia"), "Jordan": ("JOR", "Asia"),
    "Kazakhstan": ("KAZ", "Asia"), "Kuwait": ("KWT", "Asia"),
    "Lebanon": ("LBN", "Asia"), "Malaysia": ("MYS", "Asia"),
    "Maldives": ("MDV", "Asia"), "Mongolia": ("MNG", "Asia"),
    "Myanmar": ("MMR", "Asia"), "Nepal": ("NPL", "Asia"),
    "Oman": ("OMN", "Asia"), "Pakistan": ("PAK", "Asia"),
    "Philippines": ("PHL", "Asia"), "Qatar": ("QAT", "Asia"),
    "Saudi Arabia": ("SAU", "Asia"), "Singapore": ("SGP", "Asia"),
    "Sri Lanka": ("LKA", "Asia"), "Tajikistan": ("TJK", "Asia"),
    "Thailand": ("THA", "Asia"), "Turkmenistan": ("TKM", "Asia"),
    "United Arab Emirates": ("ARE", "Asia"), "Uzbekistan": ("UZB", "Asia"),
    # Europe
    "Albania": ("ALB", "Europe"), "Austria": ("AUT", "Europe"),
    "Belarus": ("BLR", "Europe"), "Belgium": ("BEL", "Europe"),
    "Bosnia and Herzegovina": ("BIH", "Europe"), "Bulgaria": ("BGR", "Europe"),
    "Croatia": ("HRV", "Europe"), "Cyprus": ("CYP", "Europe"),
    "Czechia": ("CZE", "Europe"), "Denmark": ("DNK", "Europe"),
    "Estonia": ("EST", "Europe"), "Finland": ("FIN", "Europe"),
    "France": ("FRA", "Europe"), "Germany": ("DEU", "Europe"),
    "Greece": ("GRC", "Europe"), "Hungary": ("HUN", "Europe"),
    "Iceland": ("ISL", "Europe"), "Ireland": ("IRL", "Europe"),
    "Italy": ("ITA", "Europe"), "Latvia": ("LVA", "Europe"),
    "Lithuania": ("LTU", "Europe"), "Luxembourg": ("LUX", "Europe"),
    "Malta": ("MLT", "Europe"), "Montenegro": ("MNE", "Europe"),
    "North Macedonia": ("MKD", "Europe"), "Norway": ("NOR", "Europe"),
    "Poland": ("POL", "Europe"), "Portugal": ("PRT", "Europe"),
    "Romania": ("ROU", "Europe"), "Russian Federation": ("RUS", "Europe"),
    "Serbia": ("SRB", "Europe"), "Slovenia": ("SVN", "Europe"),
    "Spain": ("ESP", "Europe"), "Sweden": ("SWE", "Europe"),
    "Switzerland": ("CHE", "Europe"), "Ukraine": ("UKR", "Europe"),
    # Latin America and the Caribbean
    "Argentina": ("ARG", "Latin America"), "Belize": ("BLZ", "Latin America"),
    "Brazil": ("BRA", "Latin America"), "Chile": ("CHL", "Latin America"),
    "Colombia": ("COL", "Latin America"), "Costa Rica": ("CRI", "Latin America"),
    "Dominican Republic": ("DOM", "Latin America"), "Ecuador": ("ECU", "Latin America"),
    "El Salvador": ("SLV", "Latin America"), "Guatemala": ("GTM", "Latin America"),
    "Guyana": ("GUY", "Latin America"), "Haiti": ("HTI", "Latin America"),
    "Honduras": ("HND", "Latin America"), "Jamaica": ("JAM", "Latin America"),
    "Mexico": ("MEX", "Latin America"), "Nicaragua": ("NIC", "Latin America"),
    "Panama": ("PAN", "Latin America"), "Paraguay": ("PRY", "Latin America"),
    "Peru": ("PER", "Latin America"), "Suriname": ("SUR", "Latin America"),
    "Trinidad and Tobago": ("TTO", "Latin America"), "Uruguay": ("URY", "Latin America"),
    # Northern America and Oceania
    "Canada": ("CAN", "Northern America"),
    "Australia": ("AUS", "Oceania"), "New Zealand": ("NZL", "Oceania"),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        sys.exit(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["year"] = df["year"].astype(int)
    df["iso3"] = df["country"].map(lambda c: COUNTRY_REF.get(c, (None, None))[0])
    df["region"] = df["country"].map(lambda c: COUNTRY_REF.get(c, (None, "Unassigned"))[1])

    missing = sorted(df.loc[df["iso3"].isna(), "country"].unique())
    if missing:
        print("Warning: no ISO3 code, so absent from the globe: " + ", ".join(missing))

    # Rank each field within its year, so a panel can say where a country sits
    # rather than only what it scores.
    for column, _label, _unit, _spec, better in RANKED:
        df[f"{column}__rank"] = df.groupby("year")[column].rank(
            ascending=(better == "down"), method="min").astype(int)

    return df


def pca_weights(df: pd.DataFrame) -> tuple[dict, dict, float]:
    """
    Recover the first principal component of the four normalised urban factors.

    This is the same derivation happiness_analysis.py used to build
    urban_happiness_index_pca: weights are the absolute PC1 loadings,
    normalised to sum to one. Done here with a plain SVD so the dashboard
    needs no scikit-learn, and recomputed from whatever file is loaded rather
    than hard coded.
    """
    matrix = df[PCA_FACTORS].to_numpy(dtype=float)
    centred = matrix - matrix.mean(axis=0)
    _u, singular, vt = np.linalg.svd(centred, full_matrices=False)

    loadings = vt[0]
    # SVD leaves the sign of a component free. Pin it so the dominant factor
    # reads positive, matching the published run.
    if loadings[np.argmax(np.abs(loadings))] < 0:
        loadings = -loadings

    weights = np.abs(loadings) / np.abs(loadings).sum()
    explained = float(singular[0] ** 2 / (singular ** 2).sum())

    by_column = {c[0]: float(w) for c, w in zip(COMPONENTS, weights)}
    loading_by_column = {c[0]: float(loading) for c, loading in zip(COMPONENTS, loadings)}
    return by_column, loading_by_column, explained


def fmt(value, spec: str) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return format(float(value), spec)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_globe(df: pd.DataFrame, year: int, measure: str, selected: str | None) -> go.Figure:
    """Orthographic projection: a globe you can spin with the mouse."""
    label = dict(GLOBE_OPTIONS)[measure]
    view = df[df["year"] == year].dropna(subset=["iso3", measure])

    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        locations=view["iso3"],
        z=view[measure],
        # Fixed to the whole panel, so a shade means the same in every year.
        zmin=float(df[measure].min()),
        zmax=float(df[measure].max()),
        colorscale=COLOURSCALE,
        marker_line_color=BG,
        marker_line_width=0.5,
        customdata=view[["country", f"{measure}__rank"]].to_numpy(),
        colorbar=dict(
            title=dict(text=f"{label}<br><span style='font-size:10.5px'>"
                            "brighter = higher</span>",
                       font=dict(color=YELLOW_MID, size=11.5), side="right"),
            thickness=11, len=0.58, x=0.99, xpad=2, outlinewidth=0,
            tickfont=dict(color=YELLOW_MID, size=11),
        ),
        hovertemplate=("<b>%{customdata[0]}</b><br>"
                       f"{label}: " + "%{z:.2f}"
                       "<br>Rank %{customdata[1]} this year"
                       "<extra></extra>"),
    ))

    # The country you picked is outlined, never recoloured, so the ramp keeps
    # its meaning.
    if selected:
        picked = view[view["country"] == selected]
        if not picked.empty:
            fig.add_trace(go.Choropleth(
                locations=picked["iso3"], z=picked[measure],
                zmin=float(df[measure].min()), zmax=float(df[measure].max()),
                colorscale=COLOURSCALE, showscale=False,
                marker_line_color=SELECTED, marker_line_width=1.8,
                customdata=picked[["country", f"{measure}__rank"]].to_numpy(),
                hovertemplate=("<b>%{customdata[0]}</b> (selected)<br>"
                               f"{label}: " + "%{z:.2f}"
                               "<br>Rank %{customdata[1]} this year<extra></extra>"),
            ))

    fig.update_layout(
        paper_bgcolor=BG,
        font=dict(family=FONT_STACK, size=12.5, color=YELLOW_MID),
        margin=dict(l=0, r=0, t=0, b=0),
        height=620,
        showlegend=False,
        # Keeps the viewer's rotation and zoom when the year changes.
        uirevision="globe",
        hoverlabel=dict(bgcolor=PANEL, bordercolor=BORDER,
                        font=dict(family=FONT_STACK, size=12.5, color=YELLOW)),
        geo=dict(
            projection=dict(type="orthographic",
                            rotation=dict(lon=10, lat=22, roll=0)),
            bgcolor=BG,
            showframe=False,
            showcoastlines=False,
            showland=True,
            landcolor=LAND,
            showocean=True,
            oceancolor=OCEAN,
            showlakes=False,
            showcountries=True,
            countrycolor=BG,
            lonaxis=dict(showgrid=True, gridcolor=GRATICULE, gridwidth=1, dtick=30),
            lataxis=dict(showgrid=True, gridcolor=GRATICULE, gridwidth=1, dtick=30),
        ),
    )
    return fig


def fig_history(df: pd.DataFrame, country: str, year: int, measure: str) -> go.Figure:
    """A small trace of the selected measure for the selected country."""
    series = df[df["country"] == country].sort_values("year")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series["year"], y=series[measure], mode="lines+markers",
        line=dict(color=YELLOW, width=2),
        marker=dict(size=8, color=YELLOW, line=dict(color=BG, width=2)),
        hovertemplate="%{x}: %{y:.2f}<extra></extra>",
    ))
    current = series[series["year"] == year]
    if not current.empty:
        fig.add_trace(go.Scatter(
            x=current["year"], y=current[measure], mode="markers",
            marker=dict(size=13, color=YELLOW_HI, line=dict(color=BG, width=2)),
            hovertemplate=f"{year}: " + "%{y:.2f}<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(family=FONT_STACK, size=11, color=YELLOW_LOW),
        margin=dict(l=2, r=6, t=6, b=2), height=104, showlegend=False,
        hoverlabel=dict(bgcolor=PANEL, bordercolor=BORDER,
                        font=dict(family=FONT_STACK, size=12, color=YELLOW)),
        xaxis=dict(showgrid=False, zeroline=False, showline=False, ticks="outside",
                   ticklen=4, tickcolor=RULE, dtick=2,
                   tickfont=dict(color=YELLOW_LOW, size=10.5)),
        yaxis=dict(showgrid=True, gridcolor=RULE, gridwidth=1, zeroline=False,
                   showline=False, ticks="", nticks=4,
                   tickfont=dict(color=YELLOW_LOW, size=10.5)),
    )
    return fig


# ---------------------------------------------------------------------------
# Panel rendering
# ---------------------------------------------------------------------------

def stat_row(label: str, unit: str, value: str, rank, total: int, weight: str = None):
    return html.Div([
        html.Div([
            html.Span(label, className="row-label"),
            html.Span(unit, className="row-unit"),
        ], className="row-name"),
        html.Div([
            html.Span(value, className="row-value"),
            html.Span(f"#{int(rank)} of {total}" if pd.notna(rank) else "",
                      className="row-rank"),
        ], className="row-figures"),
        html.Span(weight, className="row-weight") if weight else None,
    ], className="row")


def placeholder(message: str):
    return html.Div(message, className="placeholder")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Urban Happiness DSS - globe")
    parser.add_argument("--data", type=Path, default=DEFAULT_CSV,
                        help="path to happiness_analysis_results.csv")
    parser.add_argument("--port", type=int, default=8050, help="port to serve on")
    parser.add_argument("--host", default="127.0.0.1", help="host to bind")
    parser.add_argument("--debug", action="store_true", help="run Dash in debug mode")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open a browser window on start")
    return parser.parse_args(argv)


ARGS = parse_args()
DF = load_data(ARGS.data)
YEARS = sorted(DF["year"].unique().tolist())
WEIGHTS, LOADINGS, EXPLAINED = pca_weights(DF)

CSS = f"""
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: {BG};
  color: {YELLOW};
  font-family: {FONT_STACK};
  font-size: 14px;
  -webkit-font-smoothing: antialiased;
}}
.shell {{ max-width: 1560px; margin: 0 auto; padding: 14px 16px 36px; }}

/* Time bar, pinned across the top of the window */
.timebar {{
  position: sticky; top: 0; z-index: 40;
  background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px;
  padding: 12px 22px 2px; margin-bottom: 12px;
  display: grid; grid-template-columns: auto 1fr auto; gap: 24px; align-items: center;
}}
.timebar .title {{
  font-size: 14.5px; font-weight: 640; white-space: nowrap; color: {YELLOW_HI};
}}
.timebar .year {{
  font-size: 26px; font-weight: 640; min-width: 4ch; text-align: right; color: {YELLOW_HI};
}}

/* Formulae */
.formula-card {{
  background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px;
  padding: 14px 20px 16px; margin-bottom: 12px;
}}
.formula-card h2 {{
  margin: 0 0 12px; font-size: 11.5px; font-weight: 640; letter-spacing: 0.08em;
  text-transform: uppercase; color: {YELLOW_MID};
}}
.formula-block + .formula-block {{
  margin-top: 14px; padding-top: 14px; border-top: 1px solid {RULE};
}}
.formula-name {{ font-size: 13.5px; font-weight: 620; color: {YELLOW_HI}; }}
.formula {{
  margin-top: 6px; font-size: 15px; line-height: 1.75; color: {YELLOW};
}}
.formula .coef {{ color: {YELLOW_HI}; font-weight: 640; font-variant-numeric: tabular-nums; }}
.formula .term {{ color: {YELLOW}; }}
.formula .op {{ color: {YELLOW_LOW}; padding: 0 3px; }}
.formula-note {{
  margin-top: 7px; font-size: 12px; line-height: 1.6; color: {YELLOW_MID};
}}
.formula-note .caveat {{ color: {YELLOW_LOW}; }}

/* Globe flanked by its panels */
.stage {{
  display: grid; grid-template-columns: 310px minmax(0, 1fr) 330px; gap: 12px;
  align-items: start;
}}
.panel, .globe-card {{
  background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px;
}}
.panel {{ padding: 16px 18px 14px; min-height: 620px; }}
.globe-card {{ padding: 8px 8px 2px; }}

.panel h2 {{
  margin: 0; font-size: 11.5px; font-weight: 640; letter-spacing: 0.08em;
  text-transform: uppercase; color: {YELLOW_MID};
}}
.panel .country {{
  margin: 8px 0 2px; font-size: 21px; font-weight: 640; letter-spacing: -0.01em;
  color: {YELLOW_HI};
}}
.panel .place {{ font-size: 12px; color: {YELLOW_MID}; margin-bottom: 4px; }}
.group {{ margin-top: 18px; }}
.group h3 {{
  margin: 0 0 4px; font-size: 10.5px; font-weight: 640; letter-spacing: 0.07em;
  text-transform: uppercase; color: {YELLOW_LOW};
}}
.row {{
  display: grid; grid-template-columns: 1fr auto; gap: 4px 12px;
  padding: 9px 0; border-bottom: 1px solid {RULE};
}}
.row:last-child {{ border-bottom: none; }}
.row-name {{ display: flex; flex-direction: column; min-width: 0; }}
.row-label {{ color: {YELLOW}; font-size: 13.5px; }}
.row-unit {{ font-size: 11px; color: {YELLOW_LOW}; margin-top: 2px; }}
.row-figures {{ display: flex; flex-direction: column; align-items: flex-end; }}
.row-value {{
  font-size: 17px; font-weight: 640; color: {YELLOW_HI};
  font-variant-numeric: tabular-nums; line-height: 1.2;
}}
.row-rank {{
  font-size: 11px; color: {YELLOW_LOW}; margin-top: 2px;
  font-variant-numeric: tabular-nums;
}}
.row-weight {{
  grid-column: 1 / -1; font-size: 11px; color: {YELLOW_MID};
  font-variant-numeric: tabular-nums;
}}
.placeholder {{
  margin-top: 14px; font-size: 13px; line-height: 1.6; color: {YELLOW_LOW};
}}
.history-label {{
  margin-top: 16px; font-size: 10.5px; font-weight: 640; letter-spacing: 0.07em;
  text-transform: uppercase; color: {YELLOW_LOW};
}}

/* Globe shading toggle */
.toggle {{
  display: flex; gap: 8px; justify-content: center; padding: 6px 0 8px;
}}
.toggle button {{
  background: none; border: 1px solid {BORDER}; border-radius: 8px;
  color: {YELLOW_MID}; font: inherit; font-size: 12.5px; padding: 6px 14px;
  cursor: pointer;
}}
.toggle button:hover {{ color: {YELLOW_HI}; border-color: {YELLOW_LOW}; }}
.toggle button.on {{
  color: {BG}; background: {YELLOW}; border-color: {YELLOW}; font-weight: 620;
}}
.hint {{ padding: 2px 0 8px; font-size: 11.5px; color: {YELLOW_LOW}; text-align: center; }}

/* Slider */
.rc-slider-rail {{ background: {GRATICULE} !important; }}
.rc-slider-track {{ background: {YELLOW} !important; }}
.rc-slider-dot {{ background: {PANEL} !important; border-color: {YELLOW_LOW} !important; }}
.rc-slider-dot-active {{ border-color: {YELLOW} !important; }}
.rc-slider-handle {{
  background: {BG} !important; border-color: {YELLOW} !important; opacity: 1 !important;
}}
.rc-slider-mark-text {{ color: {YELLOW_LOW} !important; font-size: 11.5px !important; }}
.rc-slider-mark-text-active {{ color: {YELLOW} !important; }}

@media (max-width: 1240px) {{
  .stage {{ grid-template-columns: 1fr; }}
  .panel {{ min-height: 0; }}
}}
@media (max-width: 720px) {{
  .timebar {{ grid-template-columns: 1fr auto; row-gap: 6px; }}
  .timebar .slider-cell {{ grid-column: 1 / -1; }}
}}
"""

INDEX_TEMPLATE = """<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>__CSS__</style>
  </head>
  <body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
  </body>
</html>""".replace("__CSS__", CSS)

app = Dash(__name__, title="Urban Happiness Globe", update_title=None)
app.index_string = INDEX_TEMPLATE
server = app.server


def formula_card():
    """The two indices, written out with the weights actually in force."""
    terms = []
    for i, (column, _label, _unit, _spec, _better, _norm) in enumerate(COMPONENTS):
        if i:
            terms.append(html.Span("+", className="op"))
        terms.append(html.Span(f"{WEIGHTS[column]:.3f}", className="coef"))
        terms.append(html.Span(f"·{SHORT[column]}ₙ", className="term"))

    loading_text = ",  ".join(
        f"{SHORT[c]} {LOADINGS[c]:+.3f}" for c, *_ in COMPONENTS)

    return html.Div([
        html.H2("How each index is built"),

        html.Div([
            html.Div("Life Evaluation", className="formula-name"),
            html.Div("Cantril ladder, 0 to 10, taken as reported.", className="formula"),
            html.Div(
                "A survey measure rather than a formula: respondents place their "
                "present life on a ladder from worst possible (0) to best possible "
                "(10). It is the outcome the index below is trying to explain.",
                className="formula-note"),
        ], className="formula-block"),

        html.Div([
            html.Div("Proposed Urban Happiness Index (PCA weights)",
                     className="formula-name"),
            html.Div(["UHI", html.Span("=", className="op"), *terms],
                     className="formula"),
            html.Div([
                f"Each factor ₙ is min-max normalised to 1-10 across all countries "
                f"and years; the weighted sum is then rescaled to 1-10. Weights are the "
                f"absolute PC1 loadings normalised to sum to 1. PC1 explains "
                f"{EXPLAINED * 100:.1f}% of the variance in the four factors. ",
                html.Br(),
                html.Span(f"Loadings: {loading_text}. Because the weights take the "
                          "absolute value, all four factors enter with a positive sign, "
                          "so a higher PM2.5 reading raises the index even though the "
                          "loading is negative.", className="caveat"),
            ], className="formula-note"),
        ], className="formula-block"),
    ], className="formula-card")


app.layout = html.Div(
    [
        dcc.Store(id="selected-country", data=None),
        dcc.Store(id="globe-measure", data=GLOBE_OPTIONS[0][0]),

        html.Div([
            html.Div("Urban Happiness", className="title"),
            html.Div(
                dcc.Slider(
                    id="year-slider",
                    min=YEARS[0], max=YEARS[-1], step=None,
                    marks={y: str(y) for y in YEARS},
                    value=YEARS[-1], included=True, updatemode="mouseup",
                ),
                className="slider-cell",
            ),
            html.Div(str(YEARS[-1]), id="year-readout", className="year"),
        ], className="timebar"),

        formula_card(),

        html.Div([
            html.Div(id="panel-left", className="panel"),

            html.Div([
                html.Div([
                    html.Button(label, id={"shade": value}, n_clicks=0,
                                className="on" if i == 0 else "")
                    for i, (value, label) in enumerate(GLOBE_OPTIONS)
                ], className="toggle"),
                dcc.Graph(
                    id="globe",
                    config={"displaylogo": False, "responsive": True,
                            "scrollZoom": True,
                            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                            "toImageButtonOptions": {"format": "png", "scale": 2}},
                    style={"height": "620px"},
                ),
                html.Div("Drag to spin the globe. Click a country to fill the panels. "
                         "Grey countries have no reading in the selected year.",
                         className="hint"),
            ], className="globe-card"),

            html.Div(id="panel-right", className="panel"),
        ], className="stage"),
    ],
    className="shell",
)


@app.callback(Output("year-readout", "children"), Input("year-slider", "value"))
def show_year(year):
    return str(year)


@app.callback(
    Output("globe-measure", "data"),
    Output({"shade": GLOBE_OPTIONS[0][0]}, "className"),
    Output({"shade": GLOBE_OPTIONS[1][0]}, "className"),
    Input({"shade": GLOBE_OPTIONS[0][0]}, "n_clicks"),
    Input({"shade": GLOBE_OPTIONS[1][0]}, "n_clicks"),
    prevent_initial_call=True,
)
def choose_shading(_first, _second):
    trigger = ctx.triggered_id
    chosen = trigger.get("shade") if isinstance(trigger, dict) else GLOBE_OPTIONS[0][0]
    return chosen, ("on" if chosen == GLOBE_OPTIONS[0][0] else ""), \
        ("on" if chosen == GLOBE_OPTIONS[1][0] else "")


@app.callback(
    Output("selected-country", "data"),
    Input("globe", "clickData"),
    State("selected-country", "data"),
    prevent_initial_call=True,
)
def select_country(click, current):
    if not click or not click.get("points"):
        return no_update
    country = click["points"][0].get("customdata", [None])[0]
    if not country:
        return no_update
    # Clicking the selected country again clears it.
    return None if country == current else country


@app.callback(
    Output("globe", "figure"),
    Input("year-slider", "value"),
    Input("globe-measure", "data"),
    Input("selected-country", "data"),
)
def draw_globe(year, measure, selected):
    return fig_globe(DF, year, measure or GLOBE_OPTIONS[0][0], selected)


@app.callback(
    Output("panel-left", "children"),
    Output("panel-right", "children"),
    Input("selected-country", "data"),
    Input("year-slider", "value"),
    Input("globe-measure", "data"),
)
def fill_panels(country, year, measure):
    measure = measure or GLOBE_OPTIONS[0][0]
    total = int((DF["year"] == year).sum())

    if not country:
        empty = placeholder("Click any country on the globe to see its figures here.")
        return (
            [html.H2("Indices"), empty],
            [html.H2("Drivers"), empty],
        )

    match = DF[(DF["country"] == country) & (DF["year"] == year)]
    if match.empty:
        gap = placeholder(f"{country} has no reading for {year}. "
                          "Move the slider to a year it reports.")
        return (
            [html.H2("Indices"), html.Div(country, className="country"), gap],
            [html.H2("Drivers"), gap],
        )

    record = match.iloc[0]

    left = [
        html.H2("Indices"),
        html.Div(country, className="country"),
        html.Div(f"{record['region']} · {year} · {total} countries reporting",
                 className="place"),
        html.Div([
            stat_row(label, unit, fmt(record[column], spec),
                     record.get(f"{column}__rank"), total)
            for column, label, unit, spec, _better in INDICES
        ], className="group"),
        html.Div(f"{dict(GLOBE_OPTIONS)[measure]} over time", className="history-label"),
        dcc.Graph(figure=fig_history(DF, country, year, measure),
                  config={"displayModeBar": False},
                  style={"height": "104px", "marginTop": "4px"}),
    ]

    right = [
        html.H2("Drivers"),
        html.Div([
            html.H3("In the Urban Happiness Index"),
            *[stat_row(label, unit, fmt(record[column], spec),
                       record.get(f"{column}__rank"), total,
                       weight=f"weight {WEIGHTS[column]:.3f} "
                              f"({WEIGHTS[column] * 100:.1f}% of the index)")
              for column, label, unit, spec, _better, _norm in COMPONENTS],
        ], className="group"),
        html.Div([
            html.H3("Measured, outside the index"),
            *[stat_row(label, unit, fmt(record[column], spec),
                       record.get(f"{column}__rank"), total)
              for column, label, unit, spec, _better in CONTEXT],
        ], className="group"),
    ]
    return left, right


def main() -> None:
    url = f"http://{ARGS.host}:{ARGS.port}"
    print(f"Urban Happiness globe: {len(DF):,} rows, {DF['country'].nunique()} countries, "
          f"{YEARS[0]}-{YEARS[-1]}")
    print("Urban Happiness Index (PCA) weights: "
          + ", ".join(f"{SHORT[c]} {WEIGHTS[c]:.3f}" for c, *_ in COMPONENTS)
          + f"  (PC1 explains {EXPLAINED * 100:.1f}%)")
    print(f"Serving on {url}")
    if not ARGS.no_browser and not ARGS.debug:
        Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host=ARGS.host, port=ARGS.port, debug=ARGS.debug)


if __name__ == "__main__":
    main()
