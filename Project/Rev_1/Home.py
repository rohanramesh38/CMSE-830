import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import altair as alt
from countryinfo import CountryInfo
import plotly.express as px
import plotly.graph_objects as go
from urllib.request import urlopen
import json
from copy import deepcopy
from plotly.subplots import make_subplots
# Add a new 'iso3' column based on the mapping
from plotly.express import choropleth


from helper.data_helper import load_data_all,data_with_filtered


deathRateData=load_data_all()


st.title('Death Rate Analysis')

st.markdown("""
Analyzing the death rate of countries and the causes of death is a significant aspect of public health and epidemiology. It provides valuable insights into a country's healthcare system, overall well-being, and helps identify areas for improvement in various sectors
""")

st.header("Dataset")

st.markdown("""
This app performs analysis on the death rate and causes from different countries countries !
* **Kaggle** : [Death rate and its causes](https://www.kaggle.com/datasets/bilalwaseer/death-rate-of-countries-and-its-causes/data).
* **WHO** : [Country-level causes of death ](https://cdn.who.int/media/docs/default-source/gho-documents/global-health-estimates/ghe2019_cod_methods.pdf?sfvrsn=37bcfacc_5).
* **OWD** : [ Death by risk factor](https://ourworldindata.org/grapher/number-of-deaths-by-risk-factor )           
""")
countries=deathRateData.Entity.unique()


st.header("Countries Considered")

map_fig=choropleth(data_frame=deathRateData, locations='ISO3', color='Entity', hover_name='Entity')
map_fig.update_geos(fitbounds="locations", visible=False)
map_fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(map_fig,use_container_width=True)






# fig, ax = plt.subplots()
# ax = sns.histplot(selcted_data, bins=20, kde=True)
# st.pyplot(fig)

# fig, ax = plt.subplots()
# correlation_matrix = deathRateData.corr()
# plot=sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# st.pyplot(fig)