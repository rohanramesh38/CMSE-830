import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import altair as alt
from countryinfo import CountryInfo
from plotly.express import choropleth
from helper.data_helper import load_data_all,data_with_filtered,get_data_income_based
import plotly.express as px
import numpy as np


st.header("Ranking the Causes of Deaths by Income")

st.markdown("""
            
1. World Bank Low Income
2. World Bank Lower Middle Income
3. World Bank Upper Middle Income
4. World Bank High Income
            
""")

st.subheader("Do the less fortunate people die of different causes than others based on income?")

Income_Groups = [
    'World Bank Low Income',
    'World Bank Lower Middle Income',
    'World Bank Upper Middle Income',
    'World Bank High Income'
    ]

deathDataByIncome=get_data_income_based()




figlist=[]
for income_group in  Income_Groups:
    rank = deathDataByIncome.query( f'Entity=="{income_group}"').sort_values(by='deaths')


    rank['fraction'] = rank.deaths * 100 / rank.deaths.sum()



    fig = px.bar(rank, y='fraction', x='cause', text_auto='.2s',title="fraction of people died an the cause")

    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)

    figlist.append(fig)
   

tab1, tab2, tab3 ,tab4= st.tabs(["Low Income", "Lower Middle Income", "Upper Middle Income","High Income"])

with tab1:
   st.subheader("Low Income Countries and Death Reasons")
   st.plotly_chart(figlist[0], theme="streamlit")


with tab2:
   st.subheader("Lower Middle Income Countries and Death Reasons ")
   st.plotly_chart(figlist[1], theme="streamlit")


with tab3:
   st.subheader("Upper Middle Income Countries and Death Reasons ")
   st.plotly_chart(figlist[2], theme="streamlit")

with tab4:
   st.subheader("High Income Countries and Death Reasons")
   st.plotly_chart(figlist[3], theme="streamlit")

