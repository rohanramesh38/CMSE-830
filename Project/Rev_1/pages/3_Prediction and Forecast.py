
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import streamlit as st
import seaborn as sns
import streamlit as st
import altair as alt
# import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from helper.data_helper import load_data_all,data_with_filtered

deatRateData =load_data_all()


st.subheader("Life Expectancy Prediction")

st.write(deatRateData)


st.subheader("Regression")

models = ['LinearRegression', 'SVR', 'RandomForestRegressor', 'GradientBoostingRegressor']

cols = st.columns(len(models))
checks=[]

for i in range(0,len(models)):
    with cols[i]:
        box=st.checkbox(models[i],value=True)
        checks.append(box)

selected_models = [value for value, boolean in zip(models, checks) if boolean]



