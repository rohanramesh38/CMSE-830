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
countries=deathRateData.Entity.unique()

st.markdown("""
<style>
p {
    font-size:25px !important;
}
.ts {
    font-size:25px !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""

<div style="display: flex; align-items: center;">
  <img src="https://communicatehealth.com/wp-content/uploads/whhl/1_z6lGwbG77db9LBjzkliE9w.png" alt="Image Description" style="width: 100%; height: 100%; object-fit: cover;">
  <h1>Mortality Analysis and 
            Life Expectation Predictor</h1>
</div>
            """,unsafe_allow_html=True)


tabs=st.tabs(["what?","why?","who?","which?","How?"])

with tabs[0]:
    st.markdown("""
    <div>
    <p>This tool is designed for conducting in-depth mortality analysis and predicting life expectancy based on various determinants. It encompasses several crucial features:</p>

    <div>
        <h2>Mortality Analysis:</h2>
        <ul>
        <li>
            <h3>Demographic Trends:</h3>
            <p>Explore mortality patterns across different age groups, genders, socioeconomic backgrounds, and geographic regions to discern trends and disparities.</p>
        </li>
        <li>
            <h3>Causes of Death:</h3>
            <p>Analyze the causes and prevalence of various health and enviroinmental factors contributing to mortality rates.</p>
        </li>
        <li>
            <h3>Temporal and Spatial Trends:</h3>
            <p>Identify how mortality rates evolve over time and vary across different locations, aiding in public health interventions and policy-making.</p>
        </li>
        </ul>
    </div>
                
                """,True)
    st.markdown("""  Analyzing the death rate of countries and the causes of
                 death is a significant aspect of public health and epidemiology. 
                It provides valuable insights into a country's healthcare system, overall well-being, and helps identify areas for improvement in various sectors
                """)

    with tabs[1]:

        st.markdown("    ## Why it is imporant?")
        st.markdown("""
                    <ul>
                    <li>
    <h3>Healthcare Resource Allocation</h3>
    <p>Mortality analysis guides the allocation of healthcare resources and services. It helps in identifying areas with higher mortality rates or specific health disparities, enabling targeted interventions and better distribution of medical facilities and resources.</p>
</li>
                    <li>
    <h3>Predictive Health Planning</h3>
    <p>Predicting mortality rates and life expectancy trends based on historical data and population characteristics allows for proactive health planning. It assists in preparing for future healthcare needs, aging populations, and potential health challenges.</p>
</li>
                    <li>
    <h3>Addressing Health Disparities</h3>
    <p>Mortality analysis helps in identifying disparities in health outcomes among different demographic groups (such as age, gender, socio-economic status, etc.). It aids in developing targeted interventions to address these disparities and reduce inequities in healthcare access and outcomes.</p>
</li></ul>
    <p>Overall, mortality analysis is crucial for guiding public health initiatives, optimizing healthcare resource utilization, evaluating the effectiveness of healthcare interventions, and ultimately improving the health and well-being of populations.</p>

                    """,True)
    with tabs[2]:
        st.markdown(
        """

    ## Who cares? Why do they care?

    * **Public Health Officials**: Understanding the factors affecting death rates is crucial for public health ,officials to develop effective policies and interventions to improve health outcomes.

    * **Economists:** Variations in death rates can impact economic productivity and healthcare expenditures. Economists are interested in the economic implications of these variations.

    * **Researchers and Academics**: This project can provide valuable insights for researchers studying global health disparities and epidemiology.

    * **General Public:** People are naturally concerned about health outcomes and mortality rates, and this information can be relevant to individuals planning their lives and making health-related decisions.

    """
    )
    with tabs[3]:
        st.header(" Which Countries Considered ?")
        map_fig=choropleth(data_frame=deathRateData, locations='ISO3', color='Entity', hover_name='Entity')
        map_fig.update_geos(fitbounds="locations", visible=False)
        map_fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(map_fig,use_container_width=True)
    with tabs[4]:

        st.markdown("""
<div>
                    <h3>How are we going to achieve this</h3>

<ul class="ts">
      <li  calss="ts"> <p> <strong>Collect Data from Different Sources and Websites:</strong>Gather data from various sources, including government health departments, national statistics agencies, research papers, healthcare institutions, and global health organizations. Obtain data on death records, causes of death, demographic information, and other relevant factors contributing to mortality. </p></li>
      <li  calss="ts">  <p><strong>Exploratory Data Analysis (EDA):</strong> Perform exploratory data analysis to understand the characteristics of the collected data. This includes assessing data quality, identifying missing values, checking for outliers, and visualizing distributions and relationships between variables. EDA helps in gaining initial insights into the dataset. </p></li>    
      <li  calss="ts"> <p><strong>Analyze Trends and Patterns:</strong> Analyze mortality trends and patterns by examining the frequency and distribution of deaths over time, geographical regions, age groups, genders, and other relevant demographic factors. Identify any notable changes, disparities, or recurring patterns in mortality rates and causes of death. </p></li>
      <li  calss="ts"> <p><strong>Employ Statistical and Mathematical Modeling:</strong> Utilize statistical methods, mathematical models, and machine learning techniques to analyze mortality data. Apply regression analysis, time series analysis, survival analysis, or other statistical models to understand relationships between variables and predict mortality rates. Machine learning approaches, including decision trees, random forests, neural networks, etc., can be employed for more complex analysis and predictive modeling.</p></li>
      <li  calss="ts"> <p><strong>Predict Life Expectancy:</strong> Based on the analysis and modeling, predict life expectancy and mortality rates for different population groups or geographic regions. Develop models that take into account various factors influencing mortality, such as health indicators, socio-economic status, lifestyle behaviors, environmental factors, and medical conditions. </p></li>
      <li  calss="ts">  <p><strong>Share Information:</strong> Communicate findings, insights, and predictions derived from mortality analysis effectively. Present the results through reports, visualizations, dashboards, or presentations that are accessible and comprehensible to policymakers, healthcare professionals, researchers, and the general public. Sharing information allows for informed decision-making and the implementation of targeted interventions to improve public health outcomes.</p></li>

</ul>
                    </div>

""",True)




        st.markdown("""
        <div>
                    <h2>Dataset</h2>
        <ul>
        <li>
            <strong>World Health Organization</strong>: <a href="https://cdn.who.int/media/docs/default-source/gho-documents/global-health-estimates/ghe2019_cod_methods.pdf?sfvrsn=37bcfacc_5">Country-level causes of death</a>
        </li>
        <li>
            <strong>Our World in Data</strong>: <a href="https://ourworldindata.org/grapher/number-of-deaths-by-risk-factor">Death by risk factor</a>
        </li>
        <li>
            <strong>World Bank Data</strong>: <a href="https://datatopics.worldbank.org/world-development-indicators/the-world-by-income-and-region.html">Income by region</a>
        </li>
        <li>
            <strong>Global Health Observatory</strong>: <a href="https://www.who.int/data/gho/data/themes/mortality-and-global-health-estimates">Life expectancy and leading causes of death and disability</a>
        </li>
        </ul>
                    </div>
        """,True)
        st.markdown(
    """
    <style>
        .banner {
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #3498db;
            color: white;
            padding: 10px;
            font-size: 24px;
        }
    </style>
    """,unsafe_allow_html=True
)


#                 Analyzing the Mortality Trend of countries and the causes of death is a significant aspect of public health and epidemiology. It provides valuable insights into a country's healthcare system, overall well-being, and helps identify areas for improvement in various sectors







# **Kaggle** : [Death rate and its causes](https://www.kaggle.com/datasets/bilalwaseer/death-rate-of-countries-and-its-causes/data).


#* **Kaggle** : [Death rate and its causes](https://www.kaggle.com/datasets/bilalwaseer/death-rate-of-countries-and-its-causes/data).



col1, col2 = st.columns(2)


    






# fig, ax = plt.subplots()
# ax = sns.histplot(selcted_data, bins=20, kde=True)
# st.pyplot(fig)

# fig, ax = plt.subplots()
# correlation_matrix = deathRateData.corr()
# plot=sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# st.pyplot(fig)