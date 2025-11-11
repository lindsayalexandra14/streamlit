import streamlit as st
from streamlit_elements import elements, dashboard, mui
from streamlit_elements import plotly
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np

st.title('Python EDA: Divorce Data')

df = pd.read_csv("divorce.csv")

df["num_kids"]=df["num_kids"].fillna(0)

df = df.dropna(subset=["education_man"])

df['marriage_date'] = pd.to_datetime(df['marriage_date'])
df['divorce_date'] = pd.to_datetime(df['divorce_date'])
df['dob_man'] = pd.to_datetime(df['dob_man'])
df['dob_woman'] = pd.to_datetime(df['dob_woman'])

df["age_difference"]=abs(df["dob_man"]-df["dob_woman"])

df["days_man_older"]=df["dob_man"]-df["dob_woman"]
df["days_woman_older"]=df["dob_woman"]-df["dob_man"]
df["years_man_older"] = np.trunc(df["days_man_older"].dt.days / 365).astype(int)
df["years_woman_older"] = np.trunc(df["days_woman_older"].dt.days / 365).astype(int)

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

def plot_bars(df, col, custom_palette, sort_by='percentage', title="Title TBD", caption='Caption TBD'):
    import pandas as pd
    import plotly.express as px

    counts = df[col].value_counts()
    percentages_raw = df[col].value_counts(normalize=True) * 100

    # Sorting by category vs. percentage as option when wanting to keep categories
    # in their existing order (e.g., marriage years)
    if sort_by == 'category':
        percentages = percentages_raw.sort_index()
    elif sort_by == 'percentage':
        percentages = percentages_raw.sort_values(ascending=False)
    else:
        raise ValueError("sort_by must be 'category' or 'percentage'")

    plot_df = pd.DataFrame({
        "category": percentages.index.astype(str),  # ensure string type
        "percentage": percentages.values
    })

    # Color mapping
    categories = plot_df['category'].tolist()
    color_map = {cat: custom_palette[i % len(custom_palette)] for i, cat in enumerate(categories)}

    # Bar chart
    fig = px.bar(
        plot_df,
        x="category",
        y="percentage",
        text="percentage",
        color="category",
        color_discrete_map=color_map
    )

    # Formatting
    fig.update_layout(
        title=f'{title} Distribution',
        xaxis_title=col,
        yaxis_title="Percentage (%)",
        width=700,
        height=500,
        font=dict(family="Monospace", size=12, color="black"),
        showlegend=False,
        margin=dict(b=130)
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

    fig.add_annotation(
    text=caption,
    xref="paper", yref="paper",
    x=0, y=-0.22,
    showarrow=False,
    font=dict(size=14, color="black"),
    xanchor='left', yanchor='top',
    align="left",
    bgcolor="rgba(211, 211, 211, 0.5)",
    bordercolor="gray",
    borderpad=6,
    borderwidth=1
)

    return fig;

fig1 = plot_bars(df, "num_kids", custom_palette, title="Number of Kids", caption="Most of the (divorced) couples had 0 kids (39%) followed by<br>having 1 or 2 kids.")
    # # # Create figure with specific size
    # fig3 = px.scatter(df, x="income_man", y="income_woman", color="num_kids")
    # fig3.update_layout(width=800, height=500)  # width & height in pixels
fig2 = plot_bars(df,"education_man", custom_palette, title='Education (Man)', caption="Most of the men had a Professional-level education (higher <br>education, post-college) at 57%.")

# # Show in Streamlit
fig3 = plot_bars(df,"education_woman", custom_palette, title='Education (Woman)', caption="The women had an even higher makeup of Professional-level education<br>(higher education, post-college) at 62%.")
fig4 = plot_bars(df,"marriage_decade", custom_palette, title='Marriage Decade', caption="Over 75% of the couples were married in the '90s or '00s.")
fig5 = plot_bars(df,"marriage_year", custom_palette, title='Marriage Year', sort_by="category", caption="The highest percentage of couples were married in 1998 (5.5%).")
fig6 = plot_bars(df,"divorce_year", custom_palette, sort_by="category", title='Divorce Year', caption='The highest number of divorces among the couples occurred in <br>2011 (9.8%), with an overall peak between 2008-2011.')

# from numpy.random import default_rng as rng

# arr = rng(0).normal(1, 1, size=20)
# fig, ax = plt.subplots()
# ax.hist(arr, bins=20)

# st.pyplot(fig)



# # Create figure with specific size
# fig2 = px.scatter(df, x="income_man", y="income_woman", color="num_kids")
# fig2.update_layout(width=800, height=500)  # width & height in pixels

# # Show in Streamlit
# st.plotly_chart(fig2)

# with st.container():
#     st.write("This is inside the container")
#     fig1 = plot_bars(df, "num_kids", custom_palette, title="Number of Kids", caption="Most of the (divorced) couples had 0 kids (39%) followed by<br>having 1 or 2 kids.")
#     # # # Create figure with specific size
#     # fig3 = px.scatter(df, x="income_man", y="income_woman", color="num_kids")
#     # fig3.update_layout(width=800, height=500)  # width & height in pixels
#     fig2 = plot_bars(df,"education_man", custom_palette, title='Education (Man)', caption="Most of the men had a Professional-level education (higher <br>education, post-college) at 57%.")

#     # # Show in Streamlit
#     fig3 = plot_bars(df,"education_woman", custom_palette, title='Education (Woman)', caption="The women had an even higher makeup of Professional-level education<br>(higher education, post-college) at 62%.")
#     fig4 = plot_bars(df,"marriage_decade", custom_palette, title='Marriage Decade', caption="Over 75% of the couples were married in the '90s or '00s.")
#     fig5 = plot_bars(df,"marriage_year", custom_palette, title='Marriage Year', sort_by="category", caption="The highest percentage of couples were married in 1998 (5.5%).")
#     fig6 = plot_bars(df,"divorce_year", custom_palette, sort_by="category", title='Divorce Year', caption='The highest number of divorces among the couples occurred in <br>2011 (9.8%), with an overall peak between 2008-2011.')
      
#     st.plotly_chart(fig1)
#     st.plotly_chart(fig2)
#     st.plotly_chart(fig3)
#     st.plotly_chart(fig4)
#     st.plotly_chart(fig5)
#     st.plotly_chart(fig6)

# st.write("This is outside the container")

# row1 = st.columns(3)
# row2 = st.columns(3)

# figs = [fig1, fig2, fig3, fig4, fig5, fig6]  # Your charts
# rows = [row1, row2]

# fig_index = 0
# for r in rows:
#     for col in r:
#         if fig_index < len(figs):
#             with col:
#                 st.subheader(f"Chart {fig_index+1}")  # optional title
#                 figs[fig_index].update_layout(height=300)

#                 st.plotly_chart(figs[fig_index], use_container_width=True, key=f"chart{fig_index}")
#             fig_index += 1

import streamlit as st
from streamlit_elements import elements, dashboard, mui

# Assuming fig1, fig2, fig3, fig4, fig5, fig6 are Plotly figures already created

layout = [
    dashboard.Item("fig1", 0, 0, 3, 3),  # draggable and resizable by default
    dashboard.Item("fig2", 3, 0, 3, 3),
    dashboard.Item("fig3", 6, 0, 3, 3),
    dashboard.Item("fig4", 0, 3, 3, 3),
    dashboard.Item("fig5", 3, 3, 3, 3),
    dashboard.Item("fig6", 6, 3, 3, 3),
]

if "layout" not in st.session_state:
    st.session_state["layout"] = layout

def on_layout_change(new_layout):
    st.session_state["layout"] = new_layout

with elements("dashboard"):
    with dashboard.Grid(
        st.session_state["layout"],
        draggableHandle=".draggable",  # use header for dragging
        onLayoutChange=on_layout_change,
    ):
        for i in range(1, 7):
            key = f"fig{i}"
            title_map = {
                "fig1": "Number of Kids",
                "fig2": "Education (Man)",
                "fig3": "Education (Woman)",
                "fig4": "Marriage Decade",
                "fig5": "Marriage Year",
                "fig6": "Divorce Year"
            }
            with mui.Paper(key=key, sx={"p": 2, "height": "100%"}):
                mui.Typography(title_map[key], className="draggable", variant="h6")
                st.plotly_chart(eval(key), use_container_width=True)
