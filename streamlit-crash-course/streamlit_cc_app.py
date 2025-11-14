import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split as holdout
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import streamlit as st

st.title("Python EDA: Divorce Data")

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

def plot_histogram(df, x, nbins, xaxis_title, text):
    fig = px.histogram(
        df, 
        x=x, 
        nbins=nbins, 
        marginal="box", 
        color_discrete_sequence=['teal']
    )

    fig.update_layout(
        title=f"{xaxis_title} Distribution",
        xaxis_title=xaxis_title,
        yaxis_title="Count",
        width=600,
        height=400,
        margin=dict(b=115)
    )

    fig.update_xaxes(automargin=True)

    fig.add_annotation(
        text=text,
        xref="paper",
        yref="paper",
        x=0,
        y=-0.24,
        showarrow=False,
        font=dict(size=14, color="black"),
        xanchor='left',
        yanchor='top',
        align="left",
        bgcolor="rgba(211, 211, 211, 0.5)",
        bordercolor="gray",
        borderpad=6,
        borderwidth=1
    )  

    return fig


pairplot_columns = ['num_kids', 'marriage_duration','income_man','income_woman',
                   'age_difference',
                   'income_difference','marriage_year']


def make_pairplot(
    df,
    pairplot_columns,
    fig_title="Pairplot",
    annotation1=None,
    annotation2=None
):
    # Default annotations (if none are provided)
    if annotation1 is None:
        annotation1 = (
            "There is a strong negative correlation between marriage year and marriage duration, "
            "showing that the more recent the year the couples were<br> married, the shorter the "
            "marriage duration."
        )
    if annotation2 is None:
        annotation2 = (
            "There is a strong positive correlation between the income of the man and the income "
            "of the woman (e.g., if the income of the man is high or <br>low, so is that of the woman)"
        )

    #Create scatter matrix
    fig = px.scatter_matrix(
        df,
        dimensions=pairplot_columns,
        title=fig_title,
        height=1200,
        width=1200
    )

    # Update traces
    fig.update_traces(
        diagonal_visible=False,
        marker=dict(size=5, opacity=0.7, color='lightseagreen')
    )

    # Layout & margin
    fig.update_layout(margin=dict(b=250))

    # Add annotations
    fig.add_annotation(
        text=annotation1,
        xref="paper", yref="paper",
        x=0, y=-0.07,
        showarrow=False,
        font=dict(size=14, color="black"),
        xanchor='left', yanchor='top',
        align="left",
        bgcolor="rgba(211, 211, 211, 0.5)",
        bordercolor="gray",
        borderpad=6,
        borderwidth=1
    )

    fig.add_annotation(
        text=annotation2,
        xref="paper", yref="paper",
        x=0, y=-0.13,
        showarrow=False,
        font=dict(size=14, color="black"),
        xanchor='left', yanchor='top',
        align="left",
        bgcolor="rgba(211, 211, 211, 0.5)",
        bordercolor="gray",
        borderpad=6,
        borderwidth=1
    )

    fig.update_xaxes(tickangle=45) 
    fig.update_yaxes(tickangle=0) 

    return fig

heatmap_palette = sns.diverging_palette(
    3550,  # Hue for end 1
    44,   # Hue for end 2
    n=9,  # Number of colors
    center="light"  # midpoint lighter
)

sns.palplot(heatmap_palette)

def make_correlation_heatmap(
    df,
    heatmap_palette="coolwarm",
    fig_title="Correlation Heatmap",
    caption_text=None,
    figsize=(7, 5)
):
    # Default caption
    if caption_text is None:
        caption_text = (
            "The heatmap shows the correlations between all the numerical variables:\n"
            "\n"
            "Strong positive correlations:\n"
            "• Income (Man) & Income (Woman)\n"
            "• Number of Kids & Marriage Duration\n"
            "• Age Difference & Years Woman Older\n\n"
            "Strong negative correlations:\n"
            "• Marriage Duration & Marriage Year\n"
            "• Number of Kids & Marriage Year and Decade\n"
            "• Age Difference & Years Man Older"
        )

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Draw heatmap
    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        fmt=".2f",
        cmap=heatmap_palette,
        center=0,
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={'label': 'Correlation'}
    )

    # Formatting
    plt.title(fig_title, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Caption
    plt.figtext(
        0.04, -0.40,
        caption_text,
        fontsize=10,
        color='black',
        wrap=True,
        horizontalalignment='left',
        bbox=dict(facecolor='lightgray', alpha=0.5,
                  edgecolor='gray', boxstyle='round,pad=0.5')
    )

    return fig

columns_chi = ['education_man','education_woman', 'num_kids','marriage_decade']

df_chi = df[columns_chi].copy()

results = []

# Loop through all column pairs
for col1, col2 in combinations(df_chi.columns, 2):

    # Contingency table
    table = pd.crosstab(df_chi[col1], df_chi[col2])

    chi2, p, dof, _ = chi2_contingency(table)

    results.append({
        'feature_1': col1,
        'feature_2': col2,
        'chi2_stat': chi2,
        'p_value': p,
        'dof': dof,
        'significant': p < 0.05
    })

chi_squared_df = pd.DataFrame(results).sort_values('p_value').reset_index(drop=True)

def make_chi2_plot(
    chi_squared_df,
    fig_title="Chi-Squared Statistics for Column Pairs",
    caption_text=None
):
    # Default caption text
    if caption_text is None:
        caption_text = (
            "Chi-Squared test pairs the categorical variables to see if a significant relationship exists:\n"
            "(Note, number of kids has been used as numerical & categorical given its few unique values)\n"
            "\n"
            "• Highest signficance in relationship between Income (Man) & Income (Woman)\n"
            "• All other relationships significant except for Education (Man) & Number of Kids"
        )

    # Create label pair column
    chi_squared_df = chi_squared_df.copy()
    chi_squared_df['pair'] = chi_squared_df['feature_1'] + " × " + chi_squared_df['feature_2']

    # Dynamic figure size
    fig = plt.figure(figsize=(8, max(4, 0.3 * len(chi_squared_df))))

    # Custom color palette
    palette = sns.diverging_palette(
        280, 40, s=70, l=50,
        n=len(chi_squared_df),
        as_cmap=False
    )

    # Barplot
    ax = sns.barplot(
        data=chi_squared_df,
        x='chi2_stat',
        y='pair',
        hue='chi2_stat',
        palette=palette,
        dodge=False
    )

    # Annotate bars
    max_chi2 = chi_squared_df['chi2_stat'].max()

    for i, row in chi_squared_df.iterrows():
        stat = f"χ²={row['chi2_stat']:.0f}"
        p = f"    p={row['p_value']:.3f}"
        sig_symbol = '✔' if row['significant'] else '✘'
        sig_color = 'green' if row['significant'] else 'red'

        # Stat + p-value
        ax.text(
            row['chi2_stat'] + 0.02 * max_chi2,
            i,
            f"{stat}, {p}",
            va='center',
            ha='left',
            fontsize=9,
            color='black'
        )

        # Significance symbol
        ax.text(
            row['chi2_stat'] + 0.17 * max_chi2,
            i,
            sig_symbol,
            va='center',
            ha='left',
            fontsize=10,
            color=sig_color,
            fontweight='bold'
        )

    # Expand x-axis
    ax.set_xlim(0, max_chi2 * 1.5)

    # Caption
    plt.figtext(
        0.04, -0.21,
        caption_text,
        fontsize=10,
        color='black',
        wrap=True,
        horizontalalignment='left',
        bbox=dict(facecolor='lightgray', alpha=0.5, edgecolor='gray', boxstyle='round,pad=0.5')
    )

    # Final touches
    plt.title(fig_title, fontsize=14)
    plt.xlabel('Chi-2 Statistic')
    plt.ylabel('')
    plt.tight_layout()
    plt.legend().remove()

    return fig

fig1 = plot_bars(df, "num_kids", custom_palette, title="Number of Kids", caption="Most of the (divorced) couples had 0 kids (39%) followed by<br>having 1 or 2 kids.")
fig2 = plot_bars(df,"education_man", custom_palette, title='Education (Man)', caption="Most of the men had a Professional-level education (higher <br>education, post-college) at 57%.")


fig3 = plot_bars(df,"education_woman", custom_palette, title='Education (Woman)', caption="The women had an even higher makeup of Professional-level education<br>(higher education, post-college) at 62%.")
fig4 = plot_bars(df,"marriage_decade", custom_palette, title='Marriage Decade', caption="Over 75% of the couples were married in the '90s or '00s.")
fig5 = plot_bars(df,"marriage_year", custom_palette, title='Marriage Year', sort_by="category", caption="The highest percentage of couples were married in 1998 (5.5%).")
fig6 = plot_bars(df,"divorce_year", custom_palette, sort_by="category", title='Divorce Year', caption='The highest number of divorces among the couples occurred in <br>2011 (9.8%), with an overall peak between 2008-2011.')

fig7 = plot_histogram(df,"marriage_duration",20,"Marriage Duration",text="The median marriage duration is 8 years. The heaviest<br>concentration is between 2-5 years and the max is 33 years.")
fig8 = plot_histogram(df,"income_man",15,"Income (Man)",text="The median monthly income for the men was 5,000 dollars <br>with an IQR of 3,200-8,200 dollars and a max of ~19k.")
fig9 = plot_histogram(df,"income_woman",15,"Income (Woman)",text="The median monthly income for the women was also 5,000 dollars <br>with a bit lower Q3 (7,500 dollars) and max (~15k).")
fig10 = plot_histogram(df,"age_difference",10,"Age Difference",text="The median age difference of the couple was 2 years, with<br>most ranging from 1-4 years")
fig11 = plot_histogram(df,"years_woman_older",20,"Years Woman Older",text="The median number of years older of the woman was 1,<br> showing that in most divorced couples the woman was older.")

fig12 = make_pairplot(df, pairplot_columns, fig_title="My Pairplot")

fig13 = make_correlation_heatmap(df, heatmap_palette=heatmap_palette)

fig14 = make_chi2_plot(chi_squared_df)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Num Kids", "Education Man", "Education Woman", "Marriage Decade", "Marriage Year", "Divorce Year"]
)
for tab, fig in zip([tab1, tab2, tab3, tab4, tab5, tab6], [fig1, fig2, fig3, fig4, fig5, fig6]):
    with tab:
        st.plotly_chart(fig, use_container_width=True)

# Group 2: figs 7-11
tab7, tab8, tab9, tab10, tab11 = st.tabs(
    ["Marriage Duration","Income Man","Income Woman","Age Difference","Years Woman Older"]
)
for tab, fig in zip([tab7, tab8, tab9, tab10, tab11], [fig7, fig8, fig9, fig10, fig11]):
    with tab:
        st.plotly_chart(fig, use_container_width=True)

# Remaining figures
st.plotly_chart(fig12, use_container_width=True)  # pairplot
st.pyplot(fig13)  # correlation heatmap
st.pyplot(fig14)  # chi2 plot


# dummy variables for marriage decade (since it is now categorical)
df_encoded = pd.get_dummies(df, columns=['marriage_decade'], drop_first=True)

X_data = df[['age_difference','income_difference','num_kids','marriage_decade']].astype(float)

y_target = df['marriage_duration'].astype(float)

x_train, x_test, y_train, y_test = holdout(X_data, y_target, test_size=0.2, random_state=0)

regression = LinearRegression()
regression.fit(x_train, y_train)

predictions = regression.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(rmse)

x_train_const = sm.add_constant(x_train)
model = sm.OLS(y_train, x_train_const).fit()
print(model.summary())

vif_data = pd.DataFrame()
vif_data["feature"] = X_data.columns
vif_data["VIF"] = [variance_inflation_factor(X_data.values, i) for i in range(X_data.shape[1])]

print(vif_data)

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
