import streamlit as st

st.title('Python EDA: Divorce Data')

import pandas as pd
import numpy as np

df = pd.read_csv("divorce.csv")

import matplotlib.pyplot as plt
from numpy.random import default_rng as rng

arr = rng(0).normal(1, 1, size=20)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)

st.pyplot(fig)

import plotly.express as px

# Create figure with specific size
fig = px.scatter(df, x="income_man", y="income_woman", color="num_kids")
fig.update_layout(width=800, height=500)  # width & height in pixels

# Show in Streamlit
st.plotly_chart(fig)