import dash
from dash import html, dcc
import dash_grid_layout as dgl
import plotly.express as px
import pandas as pd
import numpy as np

# ---------------------------
# Data preparation (same as your code)
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
# Create all figures + captions
# ---------------------------
charts = [
    {"fig": plot_bars(df, "num_kids", custom_palette, title="Number of Kids"),
     "caption": "Most of the (divorced) couples had 0 kids (39%) followed by 1 or 2 kids."},
    {"fig": plot_bars(df,"education_man", custom_palette, title='Education (Man)'),
     "caption": "Most of the men had a Professional-level education (higher education, post-college) at 57%."},
    {"fig": plot_bars(df,"education_woman", custom_palette, title='Education (Woman)'),
     "caption": "The women had an even higher makeup of Professional-level education (higher education, post-college) at 62%."},
    {"fig": plot_bars(df,"marriage_decade", custom_palette, title='Marriage Decade'),
     "caption": "Over 75% of the couples were married in the '90s or '00s."},
    {"fig": plot_bars(df,"marriage_year", custom_palette, sort_by="category", title='Marriage Year'),
     "caption": "The highest percentage of couples were married in 1998 (5.5%)."},
    {"fig": plot_bars(df,"divorce_year", custom_palette, sort_by="category", title='Divorce Year'),
     "caption": "The highest number of divorces among the couples occurred in 2011 (9.8%), with an overall peak between 2008-2011."}
]

# ---------------------------
# Dash App
# ---------------------------
app = dash.Dash(__name__)

# Create draggable/resizable GridItems with caption
layout_items = []
for idx, c in enumerate(charts):
    layout_items.append(
        dgl.GridItem(
            id=f"chart{idx+1}",
            children=html.Div([
                dcc.Graph(figure=c["fig"], style={"height": "100%", "width": "100%"}),
                html.Div(c["caption"], style={"font-size":"12px", "margin-top":"5px"})
            ]),
            x=(idx%3)*4,  # 3 columns layout
            y=(idx//3)*4,
            w=4,
            h=4
        )
    )

app.layout = html.Div([
    dgl.ResponsiveGridLayout(
        children=layout_items,
        rowHeight=150,
        width=1200,
        cols=12,
        isResizable=True,
        isDraggable=True
    )
])

if __name__ == "__main__":
    app.run_server(debug=True)
