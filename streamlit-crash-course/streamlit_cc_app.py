import streamlit as st

st.title('Python EDA: Divorce Data')

import pandas as pd
import numpy as np

df = pd.read_csv("divorce.csv")

import matplotlib.pyplot as plt
from numpy.random import default_rng as rng

arr = rng(0).normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)

st.pyplot(fig)