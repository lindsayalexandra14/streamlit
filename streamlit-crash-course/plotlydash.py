import dash
from dash import html, dcc
import pandas as pd
import numpy as np
import plotly.express as px
import dash_dynamic_grid_layout as dgl

# ---------------------------
# Data preparation
# ---------------------------
df = pd.read_csv("divorce.csv")
df["num_kids"] = df["num_kids"].fillna(0)
df = df.dropna(subset=["education_man"])
df['marriage_date'] = pd.to_datetime(df['marriage_date'])
df['divorce_date'] = pd.to_datetime(df['divorce_date'])
df['dob_man'] = pd.to_datetime(df['dob_man'])
df['dob_woman'] = pd.to_datetime(df['dob_woman'])
df["age_difference"] = abs(df["dob_man"]-df["dob_woman"])
df["days_man_older"] = df["dob_man"]-df["dob_woman"]
df["days_woman_older"] = df["dob_woman"]-df["dob_man"]
df["years_man_older"] = np.trunc(df["days_man_older"].dt.days / 365).astype(int)
df["years_woman_older"] = np.trunc(df["days_woman_older"].dt.days / 365).astype(int)
df["age_difference"] = df["age_difference"].dt.days // 365
df["income_difference"] = df["income_man"]-df["income_woman"]
df["marriage_year"] = df["marriage_date"].dt.year
df["divorce_year"] = df["divorce_date"].dt.year
df["marriage_decade"] = df["marriage_date"].dt.year // 10 * 10

custom_palette = [
    '#8dd3c7','#bebada','#ab8072','#80b1d3','#b3de69',
    '#fdb462','#fccde5','#d9d9d9','#bc80bd','#17becf',
    '#aec7e8','#bbb005'
]

# ---------------------------
# Bar plot function
# ---------------------------
def plot_bars(df, col, custom_palette, sort_by='percentage', title="Title TBD"):
    counts = df[col].value_counts()
    percentages_raw = df[col].value_counts(normalize=True) * 100
    if sort_by == 'category':
        percentages = percentages_raw.sort_index()
    elif sort_by == 'percentage':
        percentages = percentages_raw.sort_values(ascending=False)
    else:
        raise ValueError("sort_by must be 'category' or 'percentage'")
    plot_df = pd.DataFrame({
        "category": percentages.index.astype(str),
        "percentage": percentages.values
    })
    categories = plot_df['category'].tolist()
    color_map = {cat: custom_palette[i % len(custom_palette)] for i, cat in enumerate(categories)}
    fig = px.bar(
        plot_df,
        x="category",
        y="percentage",
        text="percentage",
        color="category",
        color_discrete_map=color_map
    )
    fig.update_layout(
        title=title,
        xaxis_title=col,
        yaxis_title="Percentage (%)",
        margin=dict(b=80, t=40),
    )
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside',
        customdata=counts.reindex(plot_df['category']).values.reshape(-1,1),
        hovertemplate=f"{col}: %{{x}}<br>Count: %{{customdata[0]}}<br>Percentage: %{{y:.1f}}%"
    )
    return fig

# ---------------------------
# Create figures + captions
# ---------------------------
charts = [
    {"fig": plot_bars(df, "num_kids", custom_palette, title="Number of Kids"),
     "caption": "Most of the (divorced) couples had 0 kids (39%) followed by 1 or 2 kids."},
    {"fig": plot_bars(df,"education_man", custom_palette, title='Education (Man)'),
     "caption": "Most of the men had a Professional-level edu"
