#!/usr/bin/env python3
"""
Urban Happiness Decision Support System
=======================================

An interactive DSS over ``happiness_analysis_results.csv`` (the output of
``happiness_analysis.py``): 122 countries, 2011-2021 with 2013 absent, six
sustainability drivers, three composite happiness indices and the World
Happiness Report life evaluation ladder.

Every figure is built with ``plotly.graph_objects``. The year slider in the
control bar scopes every view on the page, and a play button animates it.

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
from dash import ALL, Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update

DEFAULT_CSV = Path(__file__).with_name("happiness_analysis_results.csv")

FONT_STACK = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
# Reference palette from the data-visualisation design system. Both modes are
# selected rather than flipped: the dark column holds the same hues re-stepped
# for the dark surface. Categorical slots are assigned in fixed order and never
# cycled. Caps observed here: at most six hues on the trend lines (an adjacent
# pair form), at most two on all-pairs forms such as the scatter, where
# emphasis plus direct labels carries identity instead of hue.

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "plane": "#f9f9f7",
        "ink": "#0b0b0b",
        "ink_2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "border": "rgba(11,11,11,0.10)",
        "good": "#006300",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
        "seq": ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#5598e7", "#3987e5",
                "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"],
        "seq_mid": "#86b6ef",
        "seq_deep": "#1c5cab",
        "context": "#c3c2b7",
        "div_neg": "#d03b3b",
        "div_pos": "#2a78d6",
        "div_mid": "#f0efec",
    },
    "dark": {
        "surface": "#1a1a19",
        "plane": "#0d0d0d",
        "ink": "#ffffff",
        "ink_2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "border": "rgba(255,255,255,0.10)",
        "good": "#0ca30c",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"],
        "seq": ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6",
                "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6"],
        "seq_mid": "#3987e5",
        "seq_deep": "#9ec5f4",
        "context": "#52514e",
        "div_neg": "#d03b3b",
        "div_pos": "#3987e5",
        "div_mid": "#383835",
    },
}

MAX_FOCUS = 6  # one categorical slot per focus country, never cycled


def colourscale(theme: dict) -> list:
    """Single hue sequential ramp, light to dark, as a Plotly colourscale."""
    steps = theme["seq"]
    return [[i / (len(steps) - 1), c] for i, c in enumerate(steps)]


# ---------------------------------------------------------------------------
# Measure metadata
# ---------------------------------------------------------------------------
# ``better`` records the direction of improvement. It matters: PM2.5 and the
# homicide rate are min-max normalised in happiness_analysis.py without being
# inverted, so a high pm2_5_n means dirty air, not clean air. The scenario
# builder re-orients both before weighting; the read-only views state the
# direction in their axis and colourbar titles.

CUSTOM_KEY = "custom_index"

MEASURES = {
    "urban_happiness_index": dict(
        label="Urban Happiness Index (p-value weights)", short="UHI (p-value)",
        unit="1-10", fmt=",.2f", better="up", group="Composite index",
        note="Four urban factors weighted by -log10(p) from the OLS fit.",
    ),
    "urban_happiness_index_pca": dict(
        label="Urban Happiness Index (PCA weights)", short="UHI (PCA)",
        unit="1-10", fmt=",.2f", better="up", group="Composite index",
        note="Same four urban factors, weighted by first principal component loadings.",
    ),
    "pca_happiness_index": dict(
        label="Sustainable Happiness Index (PCA)", short="SHI",
        unit="1-10", fmt=",.2f", better="up", group="Composite index",
        note="PCA over the six drivers that cleared p < 0.05. Identical to the "
             "sustainable_happiness_index column of the source file.",
    ),
    "life_evaluation": dict(
        label="Life evaluation (WHR ladder)", short="Life evaluation",
        unit="0-10", fmt=",.2f", better="up", group="Outcome",
        note="Cantril ladder score from the World Happiness Report, the variable "
             "every index is trying to explain.",
    ),
    "gdp_per_capita": dict(
        label="GDP per capita", short="GDP per capita",
        unit="int. $ PPP", fmt=",.0f", better="up", group="Driver",
        note="Purchasing power parity GDP per head.",
    ),
    "healthy_life_expectancy": dict(
        label="Healthy life expectancy", short="Healthy life exp.",
        unit="years", fmt=",.1f", better="up", group="Driver",
        note="Years of life expected in good health at birth.",
    ),
    "sdg_index_score": dict(
        label="SDG Index score", short="SDG Index",
        unit="0-100", fmt=",.1f", better="up", group="Driver",
        note="Sustainable Development Report composite score.",
    ),
    "degree_of_urbanization": dict(
        label="Degree of urbanisation", short="Urbanisation",
        unit="% urban", fmt=",.1f", better="up", group="Driver",
        note="Share of population living in urban settlements. The direction of "
             "benefit is contested, so it is treated as higher-is-more rather "
             "than higher-is-better.",
    ),
    "pm2_5": dict(
        label="PM2.5 air pollution", short="PM2.5",
        unit="ug/m3", fmt=",.1f", better="down", group="Driver",
        note="Population weighted mean fine particulate concentration. Lower is better.",
    ),
    "intentional_homicide_rate": dict(
        label="Intentional homicide rate", short="Homicide rate",
        unit="per 100k", fmt=",.2f", better="down", group="Driver",
        note="Intentional homicides per 100,000 people. Lower is better.",
    ),
    CUSTOM_KEY: dict(
        label="Custom scenario index", short="Custom index",
        unit="1-10", fmt=",.2f", better="up", group="Scenario",
        note="Built live from the driver weights set in the scenario panel, with "
             "PM2.5 and homicide re-oriented so that higher always means better.",
    ),
}

DRIVERS = [
    "gdp_per_capita",
    "healthy_life_expectancy",
    "sdg_index_score",
    "degree_of_urbanization",
    "pm2_5",
    "intentional_homicide_rate",
]

WEIGHT_PRESETS = {
    "equal": dict(gdp_per_capita=50, healthy_life_expectancy=50, sdg_index_score=50,
                  degree_of_urbanization=50, pm2_5=50, intentional_homicide_rate=50),
    "economic": dict(gdp_per_capita=90, healthy_life_expectancy=50, sdg_index_score=40,
                     degree_of_urbanization=60, pm2_5=20, intentional_homicide_rate=20),
    "environmental": dict(gdp_per_capita=20, healthy_life_expectancy=45, sdg_index_score=80,
                          degree_of_urbanization=25, pm2_5=95, intentional_homicide_rate=25),
    "social": dict(gdp_per_capita=25, healthy_life_expectancy=85, sdg_index_score=55,
                   degree_of_urbanization=30, pm2_5=30, intentional_homicide_rate=90),
}

# ---------------------------------------------------------------------------
# Country reference: ISO 3166 alpha-3 for the choropleth, region for filtering
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
    """Read the results file and add the columns the DSS needs."""
    if not csv_path.exists():
        sys.exit(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["year"] = df["year"].astype(int)
    df["iso3"] = df["country"].map(lambda c: COUNTRY_REF.get(c, (None, None))[0])
    df["region"] = df["country"].map(lambda c: COUNTRY_REF.get(c, (None, "Unassigned"))[1])

    missing = sorted(df.loc[df["iso3"].isna(), "country"].unique())
    if missing:
        print("Warning: no ISO3 code for, so absent from the map: " + ", ".join(missing))

    # Orientated 1-10 driver scores: higher always means better. The source file
    # normalises PM2.5 and homicide without inverting them, so both are flipped
    # here before they can be weighted into a scenario index.
    for driver in DRIVERS:
        source = f"{driver}_n"
        if MEASURES[driver]["better"] == "up":
            df[f"{driver}_o"] = df[source]
        else:
            df[f"{driver}_o"] = 11.0 - df[source]

    return df


def rescale_1_10(s: pd.Series) -> pd.Series:
    """Min-max a series onto 1-10 across the whole panel, so years compare."""
    lo, hi = float(s.min()), float(s.max())
    if np.isclose(lo, hi):
        return pd.Series(np.full(len(s), 5.5), index=s.index)
    return 1.0 + 9.0 * (s - lo) / (hi - lo)


def with_custom_index(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Attach the scenario index built from the supplied driver weights."""
    out = df.copy()
    total = sum(max(0.0, float(w)) for w in weights.values())
    if total <= 0:
        out[CUSTOM_KEY] = np.nan
        return out
    raw = sum(max(0.0, float(weights[d])) * out[f"{d}_o"] for d in DRIVERS) / total
    out[CUSTOM_KEY] = rescale_1_10(raw)
    return out


def rank_of(values: pd.Series, better: str) -> pd.Series:
    """Rank within the slice, 1 being the best outcome."""
    return values.rank(ascending=(better == "down"), method="min").astype(int)


def fmt(measure: str, value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return format(float(value), MEASURES[measure]["fmt"])


def axis_title(measure: str) -> str:
    meta = MEASURES[measure]
    return f"{meta['short']} ({meta['unit']})"


# ---------------------------------------------------------------------------
# Figure scaffolding
# ---------------------------------------------------------------------------

def base_layout(theme: dict, **overrides) -> dict:
    layout = dict(
        paper_bgcolor=theme["surface"],
        plot_bgcolor=theme["surface"],
        font=dict(family=FONT_STACK, size=12.5, color=theme["ink_2"]),
        margin=dict(l=8, r=16, t=16, b=8),
        hoverlabel=dict(
            bgcolor=theme["surface"],
            bordercolor=theme["border"],
            font=dict(family=FONT_STACK, size=12, color=theme["ink"]),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(color=theme["ink_2"], size=12),
            itemsizing="constant",
        ),
        showlegend=False,
        dragmode=False,
        transition=dict(duration=280, easing="cubic-in-out"),
    )
    layout.update(overrides)
    return layout


def axis_style(theme: dict, grid: bool = True, **overrides) -> dict:
    """Recessive hairline chrome: solid gridlines one shade off the surface."""
    style = dict(
        showgrid=grid,
        gridcolor=theme["grid"],
        gridwidth=1,
        zeroline=False,
        showline=False,
        ticks="outside",
        ticklen=4,
        tickcolor=theme["axis"],
        tickfont=dict(color=theme["muted"], size=11.5),
        title=dict(font=dict(color=theme["ink_2"], size=12)),
        automargin=True,
    )
    style.update(overrides)
    return style


def empty_figure(theme: dict, message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**base_layout(theme, height=None))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(color=theme["muted"], size=13),
    )
    return fig


def sparkline(values: list, theme: dict, positive: bool) -> go.Figure:
    """A bare trend line for a stat tile. The value beside it carries the number."""
    colour = theme["good"] if positive else theme["div_neg"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(values))), y=values, mode="lines",
        line=dict(color=colour, width=2, shape="spline", smoothing=0.5),
        hoverinfo="skip",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=2, b=2), height=34, showlegend=False,
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
    )
    return fig


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_map(slice_df: pd.DataFrame, measure: str, theme: dict, focus: list) -> go.Figure:
    """World choropleth: magnitude, so a single hue light to dark."""
    meta = MEASURES[measure]
    mapped = slice_df.dropna(subset=["iso3", measure])
    if mapped.empty:
        return empty_figure(theme, "No mapped countries in this slice.")

    direction = "darker = higher (worse)" if meta["better"] == "down" else "darker = higher (better)"
    ranks = rank_of(mapped[measure], meta["better"])

    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        locations=mapped["iso3"],
        z=mapped[measure],
        customdata=np.stack([mapped["country"], mapped["region"], ranks], axis=-1),
        colorscale=colourscale(theme),
        marker_line_color=theme["surface"],
        marker_line_width=0.6,
        colorbar=dict(
            title=dict(text=f"{meta['unit']}<br><span style='font-size:10.5px'>{direction}</span>",
                       font=dict(color=theme["ink_2"], size=11.5), side="right"),
            thickness=11, len=0.72, x=1.0, xpad=4, outlinewidth=0,
            tickfont=dict(color=theme["muted"], size=11),
        ),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
            f"{meta['short']}: " + "%{z:" + meta["fmt"] + "} " + meta["unit"]
            + "<br>Rank in view: %{customdata[2]}<extra></extra>"
        ),
    ))

    # Focus countries are outlined rather than recoloured, so the sequential
    # ramp keeps its meaning.
    picked = mapped[mapped["country"].isin(focus)]
    if not picked.empty:
        fig.add_trace(go.Choropleth(
            locations=picked["iso3"], z=picked[measure],
            colorscale=colourscale(theme), showscale=False,
            zmin=float(mapped[measure].min()), zmax=float(mapped[measure].max()),
            marker_line_color=theme["ink"], marker_line_width=1.6,
            customdata=np.stack([picked["country"], picked["region"],
                                 ranks[picked.index]], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b> (selected)<br>%{customdata[1]}<br>"
                f"{meta['short']}: " + "%{z:" + meta["fmt"] + "} " + meta["unit"]
                + "<br>Rank in view: %{customdata[2]}<extra></extra>"
            ),
        ))

    fig.update_layout(**base_layout(
        theme,
        height=420,
        margin=dict(l=0, r=0, t=4, b=0),
        geo=dict(
            bgcolor=theme["surface"],
            showframe=False,
            showcoastlines=False,
            showland=True,
            landcolor=theme["grid"],
            showlakes=False,
            showcountries=True,
            countrycolor=theme["surface"],
            projection_type="natural earth",
            lataxis=dict(range=[-58, 84]),
        ),
    ))
    return fig


def fig_ranking(slice_df: pd.DataFrame, measure: str, theme: dict, focus: list) -> go.Figure:
    """Top and bottom ten on one axis. One hue for context, one for emphasis."""
    meta = MEASURES[measure]
    data = slice_df.dropna(subset=[measure])
    if data.empty:
        return empty_figure(theme, "Nothing to rank in this slice.")

    ascending = meta["better"] == "down"
    ordered = data.sort_values(measure, ascending=ascending)
    ordered = ordered.assign(rank=range(1, len(ordered) + 1))

    if len(ordered) <= 20:
        shown, gap_after = ordered, None
    else:
        shown = pd.concat([ordered.head(10), ordered.tail(10)])
        gap_after = 10

    # Plotted bottom-up, so reverse to put the best at the top.
    shown = shown.iloc[::-1]
    labels = [f"{r}. {c}" for r, c in zip(shown["rank"], shown["country"])]
    is_focus = shown["country"].isin(focus).to_numpy()
    colours = [theme["series"][1] if f else theme["seq_mid"] for f in is_focus]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=shown[measure], y=labels, orientation="h",
        marker=dict(color=colours, cornerradius=4,
                    line=dict(color=theme["surface"], width=2)),
        customdata=np.stack([shown["country"], shown["region"], shown["rank"]], axis=-1),
        text=[fmt(measure, v) for v in shown[measure]],
        textposition="outside",
        textfont=dict(color=theme["ink_2"], size=11.5),
        cliponaxis=False,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
            f"{meta['short']}: " + "%{x:" + meta["fmt"] + "} " + meta["unit"]
            + "<br>Rank: %{customdata[2]} of " + str(len(ordered)) + "<extra></extra>"
        ),
    ))

    layout = base_layout(
        theme,
        height=max(340, 26 * len(shown) + 60),
        margin=dict(l=8, r=54, t=10, b=8),
        bargap=0.34,
    )
    fig.update_layout(**layout)
    fig.update_xaxes(**axis_style(theme, title=dict(
        text=axis_title(measure), font=dict(color=theme["ink_2"], size=12))))
    fig.update_yaxes(**axis_style(theme, grid=False, ticks="",
                                  tickfont=dict(color=theme["ink_2"], size=11.5)))

    if gap_after is not None:
        fig.add_hline(y=len(shown) - gap_after - 0.5, line_width=1,
                      line_color=theme["axis"])
        fig.add_annotation(
            x=1, y=len(shown) - gap_after - 0.5, xref="paper", yref="y",
            text="best above / worst below", showarrow=False, xanchor="right",
            yshift=8, font=dict(color=theme["muted"], size=10.5),
        )
    return fig


def fig_scatter(slice_df: pd.DataFrame, measure: str, theme: dict, focus: list) -> go.Figure:
    """Driver against the outcome, with an ordinary least squares fit."""
    x_measure = measure if measure != "life_evaluation" else "pca_happiness_index"
    meta = MEASURES[x_measure]
    data = slice_df.dropna(subset=[x_measure, "life_evaluation"])
    if len(data) < 3:
        return empty_figure(theme, "Too few countries for a fit.")

    x = data[x_measure].to_numpy(dtype=float)
    y = data["life_evaluation"].to_numpy(dtype=float)
    context = data[~data["country"].isin(focus)]
    picked = data[data["country"].isin(focus)]

    fig = go.Figure()

    # Fit first so the markers sit above it.
    slope, intercept = np.polyfit(x, y, 1)
    grid = np.linspace(x.min(), x.max(), 60)
    r = float(np.corrcoef(x, y)[0, 1])
    fig.add_trace(go.Scatter(
        x=grid, y=slope * grid + intercept, mode="lines", name="OLS fit",
        line=dict(color=theme["muted"], width=2),
        hovertemplate="OLS fit<br>slope %{customdata:.3f}<extra></extra>",
        customdata=np.full(len(grid), slope),
    ))
    fig.add_trace(go.Scatter(
        x=context[x_measure], y=context["life_evaluation"], mode="markers",
        name="Countries in view",
        marker=dict(size=9, color=theme["seq_mid"], opacity=0.85,
                    line=dict(color=theme["surface"], width=2)),
        customdata=np.stack([context["country"], context["region"]], axis=-1),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
            f"{meta['short']}: " + "%{x:" + meta["fmt"] + "}"
            + "<br>Life evaluation: %{y:.2f}<extra></extra>"
        ),
    ))
    if not picked.empty:
        fig.add_trace(go.Scatter(
            x=picked[x_measure], y=picked["life_evaluation"],
            mode="markers+text", name="Selected",
            marker=dict(size=13, color=theme["series"][1],
                        line=dict(color=theme["surface"], width=2)),
            text=picked["country"], textposition="top center",
            textfont=dict(color=theme["ink"], size=11.5),
            customdata=np.stack([picked["country"], picked["region"]], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                f"{meta['short']}: " + "%{x:" + meta["fmt"] + "}"
                + "<br>Life evaluation: %{y:.2f}<extra></extra>"
            ),
        ))

    fig.update_layout(**base_layout(
        theme, height=370, showlegend=True, hovermode="closest",
        margin=dict(l=8, r=16, t=40, b=8),
    ))
    fig.update_xaxes(**axis_style(theme, title=dict(
        text=axis_title(x_measure), font=dict(color=theme["ink_2"], size=12))))
    fig.update_yaxes(**axis_style(theme, title=dict(
        text="Life evaluation (0-10)", font=dict(color=theme["ink_2"], size=12))))
    fig.add_annotation(
        x=0.99, y=0.03, xref="paper", yref="paper", xanchor="right",
        text=f"r = {r:+.2f} &nbsp; n = {len(data)}",
        showarrow=False, font=dict(color=theme["ink_2"], size=12),
    )
    return fig


def fig_trend(df: pd.DataFrame, measure: str, theme: dict, focus: list,
              colour_slots: dict, regions: list, year: int) -> go.Figure:
    """Series over time for the selected countries against a regional median."""
    meta = MEASURES[measure]
    scoped = df[df["region"].isin(regions)] if regions else df
    if scoped.empty or measure not in scoped:
        return empty_figure(theme, "No data in this slice.")

    median = scoped.groupby("year")[measure].median().sort_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=median.index, y=median.to_numpy(), mode="lines",
        name="Median in view",
        line=dict(color=theme["context"], width=2),
        hovertemplate="Median in view: %{y:" + meta["fmt"] + "}<extra></extra>",
    ))

    for country in focus:
        series = scoped[scoped["country"] == country].sort_values("year")
        if series.empty:
            continue
        colour = theme["series"][colour_slots.get(country, 0) % len(theme["series"])]
        labels = [""] * len(series)
        labels[-1] = f" {country}"
        fig.add_trace(go.Scatter(
            x=series["year"], y=series[measure], mode="lines+markers+text",
            name=country,
            line=dict(color=colour, width=2),
            marker=dict(size=8, color=colour, line=dict(color=theme["surface"], width=2)),
            text=labels, textposition="middle right",
            textfont=dict(color=colour, size=11.5),
            cliponaxis=False,
            hovertemplate=f"<b>{country}</b>: " + "%{y:" + meta["fmt"] + "}<extra></extra>",
        ))

    fig.update_layout(**base_layout(
        theme, height=370, showlegend=True, hovermode="x unified",
        margin=dict(l=8, r=96, t=40, b=8),
    ))
    fig.update_xaxes(**axis_style(theme, grid=False, dtick=1,
                                  title=dict(text="", font=dict(color=theme["ink_2"]))))
    fig.update_yaxes(**axis_style(theme, title=dict(
        text=axis_title(measure), font=dict(color=theme["ink_2"], size=12))))
    fig.add_vline(x=year, line_width=1, line_color=theme["axis"])
    fig.add_annotation(
        x=year, y=1, yref="paper", text=str(year), showarrow=False,
        yshift=12, font=dict(color=theme["muted"], size=11),
    )
    return fig


def fig_profile(slice_df: pd.DataFrame, country: str, theme: dict) -> go.Figure:
    """Dumbbell: one country against the median of the countries in view."""
    if not country or slice_df.empty:
        return empty_figure(theme, "Select a country to profile.")
    row = slice_df[slice_df["country"] == country]
    if row.empty:
        return empty_figure(theme, f"No {country} record for this year.")
    row = row.iloc[0]

    names, own, med = [], [], []
    for driver in DRIVERS:
        names.append(MEASURES[driver]["short"])
        own.append(float(row[f"{driver}_o"]))
        med.append(float(slice_df[f"{driver}_o"].median()))

    fig = go.Figure()
    for i in range(len(names)):
        fig.add_trace(go.Scatter(
            x=[med[i], own[i]], y=[names[i], names[i]], mode="lines",
            line=dict(color=theme["grid"], width=3),
            hoverinfo="skip", showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=med, y=names, mode="markers", name="Median in view",
        marker=dict(size=11, color=theme["seq_mid"],
                    line=dict(color=theme["surface"], width=2)),
        hovertemplate="Median in view: %{x:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=own, y=names, mode="markers", name=country,
        marker=dict(size=13, color=theme["seq_deep"],
                    line=dict(color=theme["surface"], width=2)),
        hovertemplate=f"<b>{country}</b>: " + "%{x:.2f}<extra></extra>",
    ))

    fig.update_layout(**base_layout(
        theme, height=330, showlegend=True,
        margin=dict(l=8, r=24, t=40, b=8),
    ))
    fig.update_xaxes(**axis_style(theme, range=[0.4, 10.6], dtick=2, title=dict(
        text="Orientated score, 1-10 (higher is better)",
        font=dict(color=theme["ink_2"], size=12))))
    fig.update_yaxes(**axis_style(theme, grid=False, ticks="",
                                  tickfont=dict(color=theme["ink_2"], size=11.5)))
    return fig


def fig_scenario_rank(slice_df: pd.DataFrame, theme: dict, focus: list) -> go.Figure:
    """Top fifteen under the scenario weights."""
    data = slice_df.dropna(subset=[CUSTOM_KEY])
    if data.empty:
        return empty_figure(theme, "Give at least one driver a non-zero weight.")

    top = data.sort_values(CUSTOM_KEY, ascending=False).head(15).iloc[::-1]
    is_focus = top["country"].isin(focus).to_numpy()
    colours = [theme["series"][1] if f else theme["seq_mid"] for f in is_focus]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top[CUSTOM_KEY], y=top["country"], orientation="h",
        marker=dict(color=colours, cornerradius=4,
                    line=dict(color=theme["surface"], width=2)),
        text=[f"{v:,.2f}" for v in top[CUSTOM_KEY]],
        textposition="outside", textfont=dict(color=theme["ink_2"], size=11.5),
        cliponaxis=False,
        customdata=top["region"],
        hovertemplate="<b>%{y}</b><br>%{customdata}<br>Custom index: %{x:,.2f}<extra></extra>",
    ))
    fig.update_layout(**base_layout(
        theme, height=470, bargap=0.34, margin=dict(l=8, r=52, t=10, b=8)))
    fig.update_xaxes(**axis_style(theme, title=dict(
        text="Custom scenario index (1-10)", font=dict(color=theme["ink_2"], size=12))))
    fig.update_yaxes(**axis_style(theme, grid=False, ticks="",
                                  tickfont=dict(color=theme["ink_2"], size=11.5)))
    return fig


def fig_scenario_shift(slice_df: pd.DataFrame, baseline: str, theme: dict) -> go.Figure:
    """Rank movement against the baseline index: a diverging encoding."""
    data = slice_df.dropna(subset=[CUSTOM_KEY, baseline])
    if len(data) < 2:
        return empty_figure(theme, "Not enough countries to compare ranks.")

    base_rank = rank_of(data[baseline], MEASURES[baseline]["better"])
    scen_rank = rank_of(data[CUSTOM_KEY], "up")
    shift = (base_rank - scen_rank).rename("shift")
    frame = pd.concat([data[["country", "region"]], base_rank.rename("base"),
                       scen_rank.rename("scen"), shift], axis=1)
    frame = frame[frame["shift"] != 0]
    if frame.empty:
        return empty_figure(theme, "The scenario reproduces the baseline ranking.")

    movers = frame.reindex(frame["shift"].abs().sort_values(ascending=False).index).head(12)
    movers = movers.sort_values("shift")
    colours = [theme["div_pos"] if v > 0 else theme["div_neg"] for v in movers["shift"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=movers["shift"], y=movers["country"], orientation="h",
        marker=dict(color=colours, cornerradius=4,
                    line=dict(color=theme["surface"], width=2)),
        customdata=np.stack([movers["base"], movers["scen"], movers["region"]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>%{customdata[2]}<br>"
                       "Baseline rank %{customdata[0]} to scenario rank %{customdata[1]}"
                       "<br>Movement: %{x:+d} places<extra></extra>"),
    ))
    fig.update_layout(**base_layout(
        theme, height=470, bargap=0.34, margin=dict(l=8, r=24, t=10, b=8)))
    fig.update_xaxes(**axis_style(theme, zeroline=True, zerolinecolor=theme["axis"],
                                  zerolinewidth=1, title=dict(
        text="Places gained (right) or lost (left) against the baseline index",
        font=dict(color=theme["ink_2"], size=12))))
    fig.update_yaxes(**axis_style(theme, grid=False, ticks="",
                                  tickfont=dict(color=theme["ink_2"], size=11.5)))
    return fig


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Urban Happiness DSS")
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
REGIONS = sorted(DF["region"].unique().tolist())
COUNTRIES = sorted(DF["country"].unique().tolist())
DEFAULT_FOCUS = [c for c in ("Ireland", "Finland", "Costa Rica") if c in COUNTRIES][:MAX_FOCUS]
DEFAULT_MEASURE = "pca_happiness_index"

MEASURE_OPTIONS = [
    {"label": f"{meta['group']} - {meta['label']}", "value": key}
    for key, meta in MEASURES.items()
]

GRAPH_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}

CSS = """
:root {
  color-scheme: light;
  --surface: #fcfcfb;
  --plane: #f9f9f7;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --muted: #898781;
  --grid: #e1e0d9;
  --axis: #c3c2b7;
  --border: rgba(11,11,11,0.10);
  --accent: #2a78d6;
  --accent-soft: rgba(42,120,214,0.10);
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #1a1a19;
  --plane: #0d0d0d;
  --ink: #ffffff;
  --ink-2: #c3c2b7;
  --muted: #898781;
  --grid: #2c2c2a;
  --axis: #383835;
  --border: rgba(255,255,255,0.10);
  --accent: #3987e5;
  --accent-soft: rgba(57,135,229,0.16);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--plane);
  color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  font-size: 14px;
  -webkit-font-smoothing: antialiased;
}
.shell { max-width: 1560px; margin: 0 auto; padding: 24px 16px 56px; }

/* Header */
.masthead {
  display: flex; align-items: flex-start; justify-content: space-between;
  gap: 16px; flex-wrap: wrap; margin-bottom: 20px;
}
.masthead h1 { margin: 0; font-size: 23px; font-weight: 640; letter-spacing: -0.01em; }
.masthead p { margin: 6px 0 0; color: var(--ink-2); max-width: 74ch; line-height: 1.5; }
.ghost-btn {
  background: var(--surface); color: var(--ink-2); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 14px; font: inherit; font-size: 13px;
  cursor: pointer; white-space: nowrap;
}
.ghost-btn:hover { background: var(--accent-soft); color: var(--ink); }

/* One filter row above everything it scopes */
.controls {
  display: grid; grid-template-columns: 2fr 1.4fr 2fr auto; gap: 14px;
  align-items: end; background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 14px 16px; margin-bottom: 12px;
}
.control label {
  display: block; font-size: 11.5px; text-transform: uppercase;
  letter-spacing: 0.05em; color: var(--muted); margin-bottom: 6px;
}
.timebar {
  display: grid; grid-template-columns: auto 1fr auto; gap: 18px; align-items: center;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 12px 20px 4px; margin-bottom: 20px;
}
.year-readout { font-size: 20px; font-weight: 640; color: var(--ink); min-width: 4ch; }
.year-caption { font-size: 11.5px; color: var(--muted); }

/* Cards */
.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px 18px 12px;
}
.card h2 { margin: 0; font-size: 14.5px; font-weight: 620; color: var(--ink); }
.card .sub { margin: 4px 0 10px; font-size: 12.5px; color: var(--muted); line-height: 1.45; }

.section-title {
  margin: 26px 0 12px; font-size: 12px; font-weight: 640; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--muted);
}
.grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 16px; }
.span-4 { grid-column: span 4; }
.span-5 { grid-column: span 5; }
.span-6 { grid-column: span 6; }
.span-7 { grid-column: span 7; }
.span-8 { grid-column: span 8; }
.span-12 { grid-column: span 12; }

/* Stat tiles */
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
.tile-label {
  font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted);
}
.tile-value {
  font-size: 30px; font-weight: 620; color: var(--ink); line-height: 1.15;
  margin-top: 6px; letter-spacing: -0.015em;
}
.tile-value.small { font-size: 21px; }
.tile-sub { font-size: 12.5px; color: var(--ink-2); margin-top: 4px; }
.tile-spark { height: 34px; margin-top: 6px; }

/* Scenario weights */
.weight-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 4px 26px; }
.weight-row { padding: 2px 0 0; }
.weight-row .name {
  display: flex; justify-content: space-between; font-size: 12.5px; color: var(--ink-2);
}
.weight-row .name em { font-style: normal; color: var(--muted); font-size: 11.5px; }
.preset-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 4px 0 14px; }
.scenario-stats {
  display: flex; gap: 26px; flex-wrap: wrap; margin-top: 14px;
  padding-top: 14px; border-top: 1px solid var(--border);
}
.scenario-stats .k {
  display: block; font-size: 11.5px; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.05em;
}
.scenario-stats .v {
  display: block; font-size: 19px; font-weight: 620; color: var(--ink); margin-top: 3px;
}

/* Table view: the accessible twin of every chart above */
details.table-view { margin-top: 26px; }
details.table-view > summary {
  cursor: pointer; font-size: 13px; color: var(--ink-2); padding: 10px 0;
}
.footnote {
  margin-top: 26px; font-size: 12px; color: var(--muted); line-height: 1.6;
  max-width: 96ch;
}

/* Dash controls */
.Select-control, .Select-menu-outer, .is-open > .Select-control {
  background: var(--surface) !important; border-color: var(--border) !important;
  color: var(--ink) !important; border-radius: 8px !important;
}
.Select-value-label, .Select-placeholder,
.Select--single > .Select-control .Select-value { color: var(--ink) !important; }
.Select-menu-outer { border-radius: 8px !important; overflow: hidden; }
.VirtualizedSelectOption { background: var(--surface); color: var(--ink-2); }
.VirtualizedSelectFocusedOption { background: var(--accent-soft); color: var(--ink); }
.Select--multi .Select-value {
  background: var(--accent-soft) !important; border-color: var(--border) !important;
  color: var(--ink) !important;
}
.Select--multi .Select-value-icon { border-color: var(--border) !important; }
.rc-slider-rail { background: var(--grid) !important; }
.rc-slider-track { background: var(--accent) !important; }
.rc-slider-dot { background: var(--surface) !important; border-color: var(--axis) !important; }
.rc-slider-dot-active { border-color: var(--accent) !important; }
.rc-slider-handle {
  background: var(--surface) !important; border-color: var(--accent) !important;
  opacity: 1 !important;
}
.rc-slider-mark-text { color: var(--muted) !important; font-size: 11.5px !important; }
.rc-slider-mark-text-active { color: var(--ink-2) !important; }

@media (max-width: 1180px) {
  .controls { grid-template-columns: 1fr 1fr; }
  .kpis { grid-template-columns: repeat(2, 1fr); }
  .span-4, .span-5, .span-6, .span-7, .span-8 { grid-column: span 12; }
  .weight-grid { grid-template-columns: 1fr; }
}
@media (max-width: 640px) {
  .controls { grid-template-columns: 1fr; }
  .kpis { grid-template-columns: 1fr; }
  .timebar { grid-template-columns: 1fr; }
}
"""

INDEX_TEMPLATE = """<!DOCTYPE html>
<html data-theme="light">
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

app = Dash(__name__, title="Urban Happiness DSS", update_title=None,
           suppress_callback_exceptions=True)
app.index_string = INDEX_TEMPLATE
server = app.server


def card(title: str, subtitle: str, graph_id: str, span: str):
    return html.Div(
        [
            html.H2(title),
            html.P(subtitle, className="sub"),
            dcc.Graph(id=graph_id, config=GRAPH_CONFIG),
        ],
        className=f"card {span}",
    )


def weight_row(driver: str):
    meta = MEASURES[driver]
    direction = "lower is better, inverted" if meta["better"] == "down" else "higher is better"
    return html.Div(
        [
            html.Div([html.Span(meta["short"]), html.Em(direction)], className="name"),
            dcc.Slider(
                id={"type": "weight", "driver": driver},
                min=0, max=100, step=5, value=50,
                marks=None, tooltip={"placement": "bottom", "always_visible": False},
            ),
        ],
        className="weight-row",
    )


app.layout = html.Div(
    [
        dcc.Store(id="theme-store", data="light", storage_type="local"),
        dcc.Store(id="focus-colours", data={}),
        dcc.Interval(id="player", interval=1500, disabled=True, n_intervals=0),

        html.Div([
            html.Div([
                html.H1("Urban Happiness Decision Support System"),
                html.P(
                    "Sustainability drivers, composite indices and the World Happiness "
                    "Report ladder for 122 countries, 2011 to 2021. Move the year slider "
                    "to rescope every view; click any country on the map, ranking or "
                    "scatter to add it to the selection."
                ),
            ]),
            html.Button("Dark mode", id="theme-btn", className="ghost-btn", n_clicks=0),
        ], className="masthead"),

        # One filter row above everything it scopes.
        html.Div([
            html.Div([
                html.Label("Measure", htmlFor="measure"),
                dcc.Dropdown(id="measure", options=MEASURE_OPTIONS,
                             value=DEFAULT_MEASURE, clearable=False),
            ], className="control"),
            html.Div([
                html.Label("Regions", htmlFor="region-filter"),
                dcc.Dropdown(id="region-filter",
                             options=[{"label": r, "value": r} for r in REGIONS],
                             value=REGIONS, multi=True, placeholder="All regions"),
            ], className="control"),
            html.Div([
                html.Label(f"Countries in focus (up to {MAX_FOCUS})",
                           htmlFor="focus-countries"),
                dcc.Dropdown(id="focus-countries",
                             options=[{"label": c, "value": c} for c in COUNTRIES],
                             value=DEFAULT_FOCUS, multi=True,
                             placeholder="Select countries to track"),
            ], className="control"),
            html.Button("Clear focus", id="clear-focus", className="ghost-btn", n_clicks=0),
        ], className="controls"),

        html.Div([
            html.Button("Play", id="play-btn", className="ghost-btn", n_clicks=0),
            dcc.Slider(
                id="year-slider", min=YEARS[0], max=YEARS[-1], step=None,
                marks={y: str(y) for y in YEARS}, value=YEARS[-1],
                included=True, updatemode="mouseup",
            ),
            html.Div([
                html.Div(str(YEARS[-1]), id="year-readout", className="year-readout"),
                html.Div("reporting year", className="year-caption"),
            ]),
        ], className="timebar"),

        html.Div(id="kpi-row", className="kpis"),

        html.Div("Where things stand", className="section-title"),
        html.Div([
            card("Global distribution",
                 "Magnitude on a single hue. Countries in focus are outlined rather "
                 "than recoloured, so the ramp keeps its meaning.",
                 "fig-map", "span-8"),
            card("Ranking",
                 "Best ten and worst ten in the current slice, value labelled.",
                 "fig-ranking", "span-4"),
        ], className="grid"),

        html.Div("What moves the ladder", className="section-title"),
        html.Div([
            card("Measure against life evaluation",
                 "One point per country in the selected year, with an ordinary least "
                 "squares fit through the slice.",
                 "fig-scatter", "span-6"),
            card("Trajectories",
                 "Countries in focus against the median of the countries in view. "
                 "2013 is absent from the source data.",
                 "fig-trend", "span-6"),
        ], className="grid"),

        html.Div("Diagnose, then test a scenario", className="section-title"),
        html.Div([
            card("Driver profile",
                 "One country against the median in view, on orientated 1-10 scores "
                 "where higher always means better.",
                 "fig-profile", "span-5"),
            html.Div([
                html.H2("Scenario weights"),
                html.P(
                    "Set how much each driver should count, and the DSS rebuilds the "
                    "index live. PM2.5 and the homicide rate are inverted first, so a "
                    "higher weight always means you care more about clean air and "
                    "safer streets.",
                    className="sub"),
                html.Div([
                    html.Button("Equal", id="preset-equal", className="ghost-btn", n_clicks=0),
                    html.Button("Economic", id="preset-economic", className="ghost-btn", n_clicks=0),
                    html.Button("Environmental", id="preset-environmental", className="ghost-btn", n_clicks=0),
                    html.Button("Social", id="preset-social", className="ghost-btn", n_clicks=0),
                ], className="preset-row"),
                html.Div([weight_row(d) for d in DRIVERS], className="weight-grid"),
                html.Div(id="scenario-stats", className="scenario-stats"),
            ], className="card span-7"),
        ], className="grid"),

        html.Div([
            card("Scenario leaders",
                 "Top fifteen under the weights above, for the selected year.",
                 "fig-scenario-rank", "span-6"),
            card("Rank movement against the baseline",
                 "How far each country moves when the scenario index replaces the "
                 "measure chosen in the filter row.",
                 "fig-scenario-shift", "span-6"),
        ], className="grid"),

        html.Details([
            html.Summary("Table view of the current slice (sortable, exportable)"),
            html.Div(id="table-wrap"),
        ], className="table-view"),

        html.Div(id="footnote", className="footnote"),
    ],
    className="shell",
)


# ---------------------------------------------------------------------------
# Shared callback plumbing
# ---------------------------------------------------------------------------

SHARED_INPUTS = (
    Input("measure", "value"),
    Input("region-filter", "value"),
    Input("year-slider", "value"),
    Input("focus-countries", "value"),
    Input("focus-colours", "data"),
    Input("theme-store", "data"),
    Input({"type": "weight", "driver": ALL}, "value"),
)
WEIGHT_IDS = State({"type": "weight", "driver": ALL}, "id")


def unpack(weight_values, weight_ids, regions, focus, theme_name):
    """Normalise the shared callback arguments into usable objects."""
    weights = {i["driver"]: (v if v is not None else 0) for i, v in zip(weight_ids, weight_values)}
    for driver in DRIVERS:
        weights.setdefault(driver, 50)
    theme = THEMES.get(theme_name or "light", THEMES["light"])
    return weights, list(regions or REGIONS), list(focus or []), theme


def scoped(regions: list, weights: dict) -> pd.DataFrame:
    data = with_custom_index(DF, weights)
    return data[data["region"].isin(regions)]


# ---------------------------------------------------------------------------
# Chrome callbacks: theme, time player, focus selection
# ---------------------------------------------------------------------------

@app.callback(
    Output("theme-store", "data"),
    Input("theme-btn", "n_clicks"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def flip_theme(_, current):
    return "light" if current == "dark" else "dark"


app.clientside_callback(
    """
    function(theme) {
        var t = theme === "dark" ? "dark" : "light";
        document.documentElement.setAttribute("data-theme", t);
        return t === "dark" ? "Light mode" : "Dark mode";
    }
    """,
    Output("theme-btn", "children"),
    Input("theme-store", "data"),
)


@app.callback(
    Output("player", "disabled"),
    Output("play-btn", "children"),
    Input("play-btn", "n_clicks"),
    State("player", "disabled"),
    prevent_initial_call=True,
)
def toggle_player(_, paused):
    return (False, "Pause") if paused else (True, "Play")


@app.callback(
    Output("year-slider", "value"),
    Input("player", "n_intervals"),
    State("year-slider", "value"),
    prevent_initial_call=True,
)
def advance_year(_, year):
    index = YEARS.index(year) if year in YEARS else -1
    return YEARS[(index + 1) % len(YEARS)]


@app.callback(Output("year-readout", "children"), Input("year-slider", "value"))
def show_year(year):
    return str(year)


@app.callback(
    Output("focus-countries", "value"),
    Input("fig-map", "clickData"),
    Input("fig-ranking", "clickData"),
    Input("fig-scatter", "clickData"),
    Input("clear-focus", "n_clicks"),
    Input("focus-countries", "value"),
    prevent_initial_call=True,
)
def sync_focus(map_click, rank_click, scatter_click, _clear, current):
    trigger = ctx.triggered_id
    current = list(current or [])

    if trigger == "clear-focus":
        return []
    if trigger == "focus-countries":
        # Guard the categorical slot ceiling without recolouring survivors.
        return current[:MAX_FOCUS] if len(current) > MAX_FOCUS else no_update

    click = {"fig-map": map_click, "fig-ranking": rank_click,
             "fig-scatter": scatter_click}.get(trigger)
    if not click or not click.get("points"):
        return no_update

    point = click["points"][0]
    custom = point.get("customdata")
    if isinstance(custom, (list, tuple)) and custom:
        country = custom[0]
    elif isinstance(custom, str):
        country = custom
    else:
        country = point.get("y") or point.get("text")
    if isinstance(country, str) and ". " in country and country.split(". ")[0].isdigit():
        country = country.split(". ", 1)[1]
    if not country or country not in COUNTRIES:
        return no_update

    if country in current:
        return [c for c in current if c != country]
    if len(current) >= MAX_FOCUS:
        return no_update
    return current + [country]


@app.callback(
    Output("focus-colours", "data"),
    Input("focus-countries", "value"),
    State("focus-colours", "data"),
)
def assign_colour_slots(focus, current):
    """Colour follows the entity: a survivor of a deselection keeps its hue."""
    focus = list(focus or [])
    kept = {c: s for c, s in (current or {}).items() if c in focus}
    used = set(kept.values())
    for country in focus:
        if country in kept:
            continue
        slot = next((i for i in range(MAX_FOCUS) if i not in used), 0)
        kept[country] = slot
        used.add(slot)
    return kept


@app.callback(
    Output({"type": "weight", "driver": ALL}, "value"),
    Input("preset-equal", "n_clicks"),
    Input("preset-economic", "n_clicks"),
    Input("preset-environmental", "n_clicks"),
    Input("preset-social", "n_clicks"),
    WEIGHT_IDS,
    prevent_initial_call=True,
)
def apply_preset(_equal, _economic, _environmental, _social, weight_ids):
    trigger = ctx.triggered_id
    if not isinstance(trigger, str) or not trigger.startswith("preset-"):
        return no_update
    preset = WEIGHT_PRESETS.get(trigger.removeprefix("preset-"), WEIGHT_PRESETS["equal"])
    # Fall back to the output spec if the state arrives empty, so the sliders
    # are always written in the order Dash expects them back.
    ids = weight_ids or [spec["id"] for spec in ctx.outputs_list]
    return [preset[i["driver"]] for i in ids]


# ---------------------------------------------------------------------------
# View callbacks
# ---------------------------------------------------------------------------

def stat_tile(label: str, value: str, sub, spark=None, small: bool = False):
    children = [
        html.Div(label, className="tile-label"),
        html.Div(value, className="tile-value small" if small else "tile-value"),
    ]
    if sub:
        children.append(html.Div(sub, className="tile-sub"))
    if spark is not None:
        children.append(dcc.Graph(
            figure=spark, config={"displayModeBar": False, "staticPlot": True},
            className="tile-spark", style={"height": "34px"}))
    return html.Div(children, className="card")


@app.callback(Output("kpi-row", "children"), *SHARED_INPUTS, WEIGHT_IDS)
def update_kpis(measure, regions, year, focus, _slots, theme_name, weight_values, weight_ids):
    weights, regions, focus, theme = unpack(weight_values, weight_ids, regions, focus, theme_name)
    meta = MEASURES[measure]
    data = scoped(regions, weights)
    view = data[data["year"] == year].dropna(subset=[measure])
    if view.empty:
        return [html.Div(html.Div("No data in this slice.", className="tile-sub"),
                         className="card")]

    yearly = data.groupby("year")[measure].mean().sort_index()
    spread = data.groupby("year")[measure].quantile(0.9) - data.groupby("year")[measure].quantile(0.1)
    spread = spread.sort_index()

    prior_years = [y for y in yearly.index if y < year]
    prior = prior_years[-1] if prior_years else None
    change = yearly[year] - yearly[prior] if prior is not None else None
    improving = (change or 0) >= 0 if meta["better"] == "up" else (change or 0) <= 0

    if change is None:
        delta_text = "no earlier year in view"
    else:
        delta_text = f"{change:+,.2f} vs {prior}"

    ascending = meta["better"] == "down"
    leader = view.sort_values(measure, ascending=ascending).iloc[0]

    first_year = min(data["year"])
    start = data[data["year"] == first_year].set_index("country")[measure]
    end = view.set_index("country")[measure]
    both = start.index.intersection(end.index)
    mover_text, mover_sub = "n/a", f"no country present in both {first_year} and {year}"
    if len(both):
        movement = (end[both] - start[both])
        if meta["better"] == "down":
            movement = -movement
        best = movement.idxmax()
        mover_text = best
        mover_sub = (f"{movement[best]:+,.2f} since {first_year} "
                     f"({'improvement' if movement[best] >= 0 else 'decline'})")

    spread_prior = spread[prior] if prior is not None else None
    spread_delta = ("converging" if spread_prior is not None and spread[year] < spread_prior
                    else "diverging" if spread_prior is not None else "single year in view")

    return [
        stat_tile(
            f"Mean {meta['short']}", fmt(measure, yearly[year]), delta_text,
            sparkline(yearly.to_numpy().tolist(), theme, improving)),
        stat_tile(
            "Leader" if meta["better"] == "up" else "Lowest", leader["country"],
            f"{fmt(measure, leader[measure])} {meta['unit']} - {leader['region']}",
            small=True),
        stat_tile(
            "Spread, P90 minus P10", fmt(measure, spread[year]),
            f"{spread_delta} - {len(view)} countries in view",
            sparkline(spread.to_numpy().tolist(), theme,
                      spread_prior is not None and spread[year] < spread_prior)),
        stat_tile("Biggest improvement", mover_text, mover_sub, small=True),
    ]


@app.callback(Output("fig-map", "figure"), *SHARED_INPUTS, WEIGHT_IDS)
def update_map(measure, regions, year, focus, _slots, theme_name, weight_values, weight_ids):
    weights, regions, focus, theme = unpack(weight_values, weight_ids, regions, focus, theme_name)
    view = scoped(regions, weights)
    return fig_map(view[view["year"] == year], measure, theme, focus)


@app.callback(Output("fig-ranking", "figure"), *SHARED_INPUTS, WEIGHT_IDS)
def update_ranking(measure, regions, year, focus, _slots, theme_name, weight_values, weight_ids):
    weights, regions, focus, theme = unpack(weight_values, weight_ids, regions, focus, theme_name)
    view = scoped(regions, weights)
    return fig_ranking(view[view["year"] == year], measure, theme, focus)


@app.callback(Output("fig-scatter", "figure"), *SHARED_INPUTS, WEIGHT_IDS)
def update_scatter(measure, regions, year, focus, _slots, theme_name, weight_values, weight_ids):
    weights, regions, focus, theme = unpack(weight_values, weight_ids, regions, focus, theme_name)
    view = scoped(regions, weights)
    return fig_scatter(view[view["year"] == year], measure, theme, focus)


@app.callback(Output("fig-trend", "figure"), *SHARED_INPUTS, WEIGHT_IDS)
def update_trend(measure, regions, year, focus, slots, theme_name, weight_values, weight_ids):
    weights, regions, focus, theme = unpack(weight_values, weight_ids, regions, focus, theme_name)
    return fig_trend(with_custom_index(DF, weights), measure, theme, focus,
                     slots or {}, regions, year)


@app.callback(Output("fig-profile", "figure"), *SHARED_INPUTS, WEIGHT_IDS)
def update_profile(measure, regions, year, focus, _slots, theme_name, weight_values, weight_ids):
    weights, regions, focus, theme = unpack(weight_values, weight_ids, regions, focus, theme_name)
    view = scoped(regions, weights)
    view = view[view["year"] == year]
    if view.empty:
        return empty_figure(theme, "No data in this slice.")
    if focus:
        country = focus[0]
    else:
        ascending = MEASURES[measure]["better"] == "down"
        ranked = view.dropna(subset=[measure]).sort_values(measure, ascending=ascending)
        country = ranked.iloc[0]["country"] if not ranked.empty else None
    return fig_profile(view, country, theme)


@app.callback(Output("fig-scenario-rank", "figure"), *SHARED_INPUTS, WEIGHT_IDS)
def update_scenario_rank(measure, regions, year, focus, _slots, theme_name, weight_values, weight_ids):
    weights, regions, focus, theme = unpack(weight_values, weight_ids, regions, focus, theme_name)
    view = scoped(regions, weights)
    return fig_scenario_rank(view[view["year"] == year], theme, focus)


@app.callback(Output("fig-scenario-shift", "figure"), *SHARED_INPUTS, WEIGHT_IDS)
def update_scenario_shift(measure, regions, year, focus, _slots, theme_name, weight_values, weight_ids):
    weights, regions, focus, theme = unpack(weight_values, weight_ids, regions, focus, theme_name)
    baseline = measure if measure != CUSTOM_KEY else "life_evaluation"
    view = scoped(regions, weights)
    return fig_scenario_shift(view[view["year"] == year], baseline, theme)


@app.callback(Output("scenario-stats", "children"), *SHARED_INPUTS, WEIGHT_IDS)
def update_scenario_stats(measure, regions, year, focus, _slots, theme_name, weight_values, weight_ids):
    weights, regions, focus, theme = unpack(weight_values, weight_ids, regions, focus, theme_name)
    view = scoped(regions, weights)
    view = view[view["year"] == year].dropna(subset=[CUSTOM_KEY, "life_evaluation"])
    if len(view) < 3:
        return html.Div([html.Span("Scenario", className="k"),
                         html.Span("Not enough countries", className="v")])

    r = float(np.corrcoef(view[CUSTOM_KEY], view["life_evaluation"])[0, 1])
    rho = float(np.corrcoef(view[CUSTOM_KEY].rank(), view["life_evaluation"].rank())[0, 1])
    leader = view.sort_values(CUSTOM_KEY, ascending=False).iloc[0]

    baseline = measure if measure != CUSTOM_KEY else "life_evaluation"
    base_r = float(np.corrcoef(view[baseline], view["life_evaluation"])[0, 1])

    def block(key, value):
        return html.Div([html.Span(key, className="k"), html.Span(value, className="v")])

    return [
        block("Scenario leader", leader["country"]),
        block("Fit to life evaluation", f"r = {r:+.2f}"),
        block("Rank agreement", f"rho = {rho:+.2f}"),
        block(f"Baseline {MEASURES[baseline]['short']}", f"r = {base_r:+.2f}"),
    ]


@app.callback(Output("table-wrap", "children"), *SHARED_INPUTS, WEIGHT_IDS)
def update_table(measure, regions, year, focus, _slots, theme_name, weight_values, weight_ids):
    weights, regions, focus, theme = unpack(weight_values, weight_ids, regions, focus, theme_name)
    meta = MEASURES[measure]
    view = scoped(regions, weights)
    view = view[view["year"] == year].dropna(subset=[measure]).copy()
    if view.empty:
        return html.Div("No rows in this slice.", className="tile-sub")

    view["rank"] = rank_of(view[measure], meta["better"])
    columns = ["rank", "country", "region", measure, CUSTOM_KEY, "life_evaluation"] + DRIVERS
    columns = list(dict.fromkeys(columns))
    table = view.sort_values("rank")[columns].round(3)

    headers = {
        "rank": "Rank", "country": "Country", "region": "Region",
        "life_evaluation": "Life evaluation",
    }
    headers.update({k: MEASURES[k]["short"] for k in MEASURES if k in columns})
    headers["rank"] = "Rank"

    return dash_table.DataTable(
        data=table.to_dict("records"),
        columns=[{"name": headers.get(c, c), "id": c} for c in columns],
        sort_action="native",
        filter_action="native",
        page_size=20,
        export_format="csv",
        export_headers="display",
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": theme["surface"], "color": theme["ink_2"],
            "fontWeight": "620", "border": "none",
            "borderBottom": f"1px solid {theme['axis']}",
            "fontFamily": FONT_STACK, "fontSize": "12.5px",
        },
        style_cell={
            "backgroundColor": theme["surface"], "color": theme["ink_2"],
            "border": "none", "borderBottom": f"1px solid {theme['grid']}",
            "fontFamily": FONT_STACK, "fontSize": "12.5px",
            "fontVariantNumeric": "tabular-nums",
            "padding": "8px 12px", "textAlign": "right",
        },
        style_cell_conditional=[
            {"if": {"column_id": c}, "textAlign": "left"} for c in ("country", "region")
        ],
        style_data_conditional=[
            {"if": {"filter_query": "{country} = " + f'"{c}"'},
             "backgroundColor": theme["grid"], "color": theme["ink"]}
            for c in focus
        ],
        style_filter={
            "backgroundColor": theme["surface"], "color": theme["ink_2"],
            "border": "none", "borderBottom": f"1px solid {theme['grid']}",
        },
    )


@app.callback(Output("footnote", "children"), Input("measure", "value"))
def update_footnote(measure):
    meta = MEASURES[measure]
    return [
        html.Div([html.Strong(f"{meta['label']}. "), meta["note"]]),
        html.Div(
            "Source: happiness_analysis_results.csv, produced by happiness_analysis.py "
            "from the World Happiness Report, World Bank GDP per capita, WHO healthy "
            "life expectancy, PM2.5 exposure, UNODC homicide rates, degree of "
            "urbanisation and the SDG Index. 2013 is absent from the panel.",
            style={"marginTop": "8px"}),
        html.Div(
            "Caveat carried from the source file: the _n columns are min-max "
            "normalised without inverting PM2.5 or the homicide rate, so both enter "
            "the published composite indices with a positive sign even though lower "
            "values are better. The scenario builder on this page re-orients them "
            "before weighting; the published indices above are shown as produced. "
            "sustainable_happiness_index is an exact copy of pca_happiness_index in "
            "this run, so only one of the two is offered.",
            style={"marginTop": "8px"}),
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    url = f"http://{ARGS.host}:{ARGS.port}"
    print(f"Urban Happiness DSS: {len(DF):,} rows, {DF['country'].nunique()} countries, "
          f"{YEARS[0]}-{YEARS[-1]}")
    print(f"Serving on {url}")
    if not ARGS.no_browser and not ARGS.debug:
        Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host=ARGS.host, port=ARGS.port, debug=ARGS.debug)


if __name__ == "__main__":
    main()
