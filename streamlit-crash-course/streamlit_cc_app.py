import streamlit as st

st.title('Python EDA: Divorce Data')

import pandas as pd
import numpy as np

df = pd.read_csv("divorce.csv")

df["num_kids"]=df["num_kids"].fillna(0)

df = df.dropna(subset=["education_man"])

df['marriage_date'] = pd.to_datetime(df['marriage_date'])
df['divorce_date'] = pd.to_datetime(df['divorce_date'])
df['dob_man'] = pd.to_datetime(df['dob_man'])
df['dob_woman'] = pd.to_datetime(df['dob_woman'])

df["age_difference"]=abs(df["dob_man"]-df["dob_woman"])
df["years_man_older"]=df["dob_man"]-df["dob_woman"]
df["years_woman_older"]=df["dob_woman"]-df["dob_man"]
df["years_man_older"] = df["years_man_older"].dt.days // 365
df["years_woman_older"] = df["years_woman_older"].dt.days // 365
df["age_difference"] = df["age_difference"].dt.days // 365
df["income_difference"]=df["income_man"]-df["income_woman"]
df["marriage_year"]=df["marriage_date"].dt.year
df["divorce_year"]=df["divorce_date"].dt.year
df["marriage_decade"]=df["marriage_date"].dt.year // 10 * 10

custom_palette = [
    '#8dd3c7',
    '#bebada',
    '#ab8072',
    '#80b1d3',
    '#b3de69',
    '#fdb462',
    '#fccde5',
    '#d9d9d9',
    '#bc80bd',
    '#17becf',
    '#aec7e8',
    '#bbb005']

def plot_bars(df, col, custom_palette):
    import pandas as pd
    import plotly.express as px

    # Counts and percentages
    counts = df[col].value_counts()
    percentages = (df[col].value_counts(normalize=True) * 100).sort_index()  # keep ascending order

    # Prepare DataFrame
    plot_df = pd.DataFrame({
        "category": percentages.index.astype(str),  # ensure string type
        "percentage": percentages.values
    })

    # Map colors to categories
    categories = plot_df['category'].tolist()
    color_map = {cat: custom_palette[i % len(custom_palette)] for i, cat in enumerate(categories)}

    # Create figure
    fig = px.bar(
        plot_df,
        x="category",
        y="percentage",
        text="percentage",
        color="category",
        color_discrete_map=color_map
    )

    # Layout
    fig.update_layout(
        title=f"{col} Distribution (%)",
        xaxis_title=col,
        yaxis_title="Percentage (%)",
        width=700,
        height=500,
        font=dict(family="Monospace", size=12, color="black"),
        showlegend=False  # hide legend if desired,
)

    # Hover and text formatting
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside',
        customdata=counts.reindex(plot_df['category']).values.reshape(-1,1),
        hovertemplate=f"{col}: %{{x}}<br>" +
                      "Count: %{customdata[0]}<br>" +
                      "Percentage: %{y:.1f}%"
    )

    return fig


import matplotlib.pyplot as plt
# from numpy.random import default_rng as rng

# arr = rng(0).normal(1, 1, size=20)
# fig, ax = plt.subplots()
# ax.hist(arr, bins=20)

# st.pyplot(fig)

import plotly.express as px

# # Create figure with specific size
# fig2 = px.scatter(df, x="income_man", y="income_woman", color="num_kids")
# fig2.update_layout(width=800, height=500)  # width & height in pixels

# # Show in Streamlit
# st.plotly_chart(fig2)

with st.container():
    st.write("This is inside the container")
    plot_bars(df,"num_kids", custom_palette)
    # # # Create figure with specific size
    # fig3 = px.scatter(df, x="income_man", y="income_woman", color="num_kids")
    # fig3.update_layout(width=800, height=500)  # width & height in pixels

    # # Show in Streamlit
    # st.plotly_chart(fig3)

st.write("This is outside the container")