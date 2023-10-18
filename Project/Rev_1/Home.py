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
st.set_page_config(layout="wide")


deathRateData=load_data_all()


st.title('Death Rate Analysis')

st.markdown("""
Analyzing the death rate of countries and the causes of death is a significant aspect of public health and epidemiology. It provides valuable insights into a country's healthcare system, overall well-being, and helps identify areas for improvement in various sectors
""")




# **Kaggle** : [Death rate and its causes](https://www.kaggle.com/datasets/bilalwaseer/death-rate-of-countries-and-its-causes/data).
st.markdown("""
## Dataset
* **World Health Organization** : [Country-level causes of death ](https://cdn.who.int/media/docs/default-source/gho-documents/global-health-estimates/ghe2019_cod_methods.pdf?sfvrsn=37bcfacc_5).
* **Our World in Data** : [ Death by risk factor](https://ourworldindata.org/grapher/number-of-deaths-by-risk-factor )
* **World Bank Data** : [Income by region](https://datatopics.worldbank.org/world-development-indicators/the-world-by-income-and-region.html)
* **Global Health Observatory** : [Life expectancy and leading causes of death and disability](https://www.who.int/data/gho/data/themes/mortality-and-global-health-estimates)     

""")
countries=deathRateData.Entity.unique()

#* **Kaggle** : [Death rate and its causes](https://www.kaggle.com/datasets/bilalwaseer/death-rate-of-countries-and-its-causes/data).

st.markdown(
    """

## Who cares? Why do they care?

* **Public Health Officials**: Understanding the factors affecting death rates is crucial for public health ,officials to develop effective policies and interventions to improve health outcomes.

* **Economists:** Variations in death rates can impact economic productivity and healthcare expenditures. Economists are interested in the economic implications of these variations.

* **Researchers and Academics**: This project can provide valuable insights for researchers studying global health disparities and epidemiology.

* **General Public:** People are naturally concerned about health outcomes and mortality rates, and this information can be relevant to individuals planning their lives and making health-related decisions.

"""
)

col1, col2 = st.columns(2)


    


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