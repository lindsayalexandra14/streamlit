# divorce_dash.py

import dash
from dash import dcc, html
import dash_draggable
import pandas as pd
import plotly.express as px
import numpy as np

# -----------------------------
# Initialize the Dash app
# -----------------------------
app = dash.Dash(__name__)

# -----------------------------
# Load and prepare data
# -----------------------------
df = pd.read_csv("divorce.csv")

df["num_kids"] = df["num_kids"].fillna(0)
df = df.dropna(subset=["education_man"])

df['marriage_date'] = pd.to_datetime(df['marriage_date'])
df['divorce_date'] = pd.to_datetime(df['divorce_date'])
df['dob_man'] = pd.to_datetime(df['dob_man'])
df['dob_woman'] = pd.to_datetime(df['dob_woman'])

df["age_difference"] = abs(df["dob_man"] - df["dob_woman"])
df["days_man_older"] = df["dob_man"] - df["dob_woman"]
df["days_woman_older"] = df["dob_woman"] - df["dob_man"]
df["years_man_older"] = np.trunc(df["days_man_older"].dt.days / 365).astype(int)
df["years_woman_older"] = np.trunc(df["days_woman_older"].dt.days / 365).astype(int)
df["age_difference"] = df["age_difference"].dt.days // 365
df["income_difference"] = df["income_man"] - df["income_woman"]
df["marriage_year"] = df["marriage_date"].dt.year
df["divorce_year"] = df["divorce_date"].dt.year
df["marriage_decade"] = df["marriage_date"].dt.year // 10 * 10

# -----------------------------
# Color palette
# -----------------------------
custom_palette = [
    '#8dd3c7', '#bebada', '#ab8072', '#80b1d3', '#b3de69',
    '#fdb462', '#fccde5', '#d9d9d9', '#bc80bd', '#17becf',
    '#aec7e8', '#bbb005'
]

# -----------------------------
# Plotly chart helper function
# -----------------------------
def plot_bars(df, col, custom_palette, sort_by='percentage', title="Title TBD", caption=''):
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
        title=f'{title}',
        xaxis_title=col,
        yaxis_title="Percentage (%)",
        width=650,
        height=450,
        font=dict(family="Monospace", size=12, color="black"),
        showlegend=False,
        margin=dict(b=130)
    )

    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside',
        customdata=counts.reindex(plot_df['category']).values.reshape(-1, 1),
        hovertemplate=f"{col}: %{{x}}<br>Count: %{{customdata[0]}}<br>Percentage: %{{y:.1f}}%"
    )

    fig.add_annotation(
        text=caption,
        xref="paper", yref="paper",
        x=0, y=-0.22,
        showarrow=False,
        font=dict(size=13, color="black"),
        xanchor='left', yanchor='top',
        align="left",
        bgcolor="rgba(211,211,211,0.4)",
        bordercolor="gray",
        borderpad=6,
        borderwidth=1
    )

    return fig


# -----------------------------
# Create figures
# -----------------------------
figs = {
    "fig1": plot_bars(df, "num_kids", custom_palette, title="Number of Kids",
                      caption="Most divorced couples had 0 kids (39%), followed by 1–2 kids."),
    "fig2": plot_bars(df, "education_man", custom_palette, title="Education (Man)",
                      caption="Most men had professional-level education (57%)."),
    "fig3": plot_bars(df, "education_woman", custom_palette, title="Education (Woman)",
                      caption="Women had slightly higher professional education (62%)."),
    "fig4": plot_bars(df, "marriage_decade", custom_palette, title="Marriage Decade",
                      caption="Over 75% of couples were married in the 1990s or 2000s."),
    "fig5": plot_bars(df, "marriage_year", custom_palette, sort_by="category",
                      title="Marriage Year", caption="Most marriages occurred around 1998 (5.5%)."),
    "fig6": plot_bars(df, "divorce_year", custom_palette, sort_by="category",
                      title="Divorce Year", caption="Peak divorces between 2008–2011 (max 9.8%).")
}

# -----------------------------
# Draggable Layout
# -----------------------------
layout_items = [
    dash_draggable.ResponsiveGridLayoutItem(
        id="fig1", x=0, y=0, w=3, h=3, children=[dcc.Graph(figure=figs["fig1"])], isDraggable=True, isResizable=True),
    dash_draggable.ResponsiveGridLayoutItem(
        id="fig2", x=3, y=0, w=3, h=3, children=[dcc.Graph(figure=figs["fig2"])], isDraggable=True, isResizable=True),
    dash_draggable.ResponsiveGridLayoutItem(
        id="fig3", x=6, y=0, w=3, h=3, children=[dcc.Graph(figure=figs["fig3"])], isDraggable=True, isResizable=True),
    dash_draggable.ResponsiveGridLayoutItem(
        id="fig4", x=0, y=3, w=3, h=3, children=[dcc.Graph(figure=figs["fig4"])], isDraggable=True, isResizable=True),
    dash_draggable.ResponsiveGridLayoutItem(
        id="fig5", x=3, y=3, w=3, h=3, children=[dcc.Graph(figure=figs["fig5"])], isDraggable=True, isResizable=True),
    dash_draggable.ResponsiveGridLayoutItem(
        id="fig6", x=6, y=3, w=3, h=3, children=[dcc.Graph(figure=figs["fig6"])], isDraggable=True, isResizable=True),
]

app.layout = html.Div([
    html.H1("📊 Divorce Data Dashboard", style={"textAlign": "center"}),
    dash_draggable.ResponsiveGridLayout(
        id="grid",
        children=layout_items,
        cols={'lg': 12, 'md': 10, 'sm': 6, 'xs': 4, 'xxs': 2},
        rowHeight=160,
        isDraggable=True,
        isResizable=True,
        style={"background": "#f7f7f7", "padding": "10px"}
    )
])

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    print("✅ Dash app starting! Use the forwarded port in Codespaces to view it.")
    app.run(debug=True, host="0.0.0.0", port=8050)
