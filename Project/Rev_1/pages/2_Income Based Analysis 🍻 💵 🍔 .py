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
st.set_page_config(layout="wide")
st.header("Ranking the Causes of Deaths by Income")

st.markdown("""
            
1. World Bank Low Income
2. World Bank Lower Middle Income
3. World Bank Upper Middle Income
4. World Bank High Income
            
""")

st.markdown("""
Given that the primary emphasis of the data is on the underlying factors contributing to long-term mortality, examining the data with income appeared to offer the potential for uncovering unique correlations. Consequently, below visualizations are dedicated to exploring the insights that the data can provide regarding the influence of income and relations to the mortality.
            """)


st.divider()

st.subheader("Do the less fortunate people die of different causes than others based on income?")

Income_Groups = [
    'World Bank Low Income',
    'World Bank Lower Middle Income',
    'World Bank Upper Middle Income',
    'World Bank High Income'
    ]

dreason={}

deathDataByIncome=get_data_income_based()
deathRateData=load_data_all("parsed_1")
#st.write(deathDataByIncome)

figlist=[]
for income_group in  Income_Groups:
    rank = deathDataByIncome.query( f'Entity=="{income_group}"').sort_values(by='deaths')

    rank['Fraction of Deaths'] = rank.deaths * 100 / rank.deaths.sum(numeric_only=True)

    rank=rank.sort_values(by='Fraction of Deaths', ascending=False)
    dreason[income_group]=rank['cause'].iloc[0]

    fig = px.bar(rank, y='Fraction of Deaths', x='cause', text_auto='.2s',title=income_group)

    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)

    figlist.append(fig)
   

tab1, tab2=st.columns(2)

tab3 ,tab4= st.columns(2)
with tab1:
   st.plotly_chart(figlist[0], theme="streamlit",use_container_width=True)

with tab2:
   st.plotly_chart(figlist[1], theme="streamlit",use_container_width=True)


with tab3:
   st.plotly_chart(figlist[2], theme="streamlit",use_container_width=True)

with tab4:
   st.plotly_chart(figlist[3], theme="streamlit",use_container_width=True)


st.markdown("""
            
Form the graph you can infer
1. That individuals with lower socioeconomic status experience a broader range of health issues compared to those with higher income.
2. It is evident from the graph that certain health conditions tend to disproportionately affect wealthier individuals, while distinct categories of ailments are more prevalent among those with lower income.
""")

st.markdown(f"""The major reason of death in  Low Income region is **{dreason["World Bank Low Income"]}**""")
st.markdown(f"""The major reason of death in  High Income region is **{dreason["World Bank High Income"]}**""")

st.divider()

st.markdown("### Profiling Causes of Death - Indicator of Income?")

affects_rich =['High systolic blood pressure','Diet high in sodium ','Diet low in whole grains','Alochol use','Diet low in fruits','Secondhand smoke','Diet low in nuts and seeds','Diet low in Vegetables','Low physical activity','Smoking','High fasting plasma glucose','High body mass index','Drug use','Iron deficiency']

affects_poor = [x for x in list(deathRateData.columns)[3:-3] if x not in affects_rich]


def profile(x):
    poor = x[affects_poor].sum()
    rich = x[affects_rich].sum()
    return 100 * rich / (rich + poor)

deathRateData = deathRateData.groupby( ['Entity','Code'], as_index=False).agg(np.sum)

deathRateData['profile'] = deathRateData.apply( profile, axis=1)

url='https://raw.githubusercontent.com/rohanramesh38/CMSE-830/main/Project/Rev_1/CLASS_1.csv'

df_meta = pd.read_csv(url)

# st.write(deathRateData)
# st.write(df_meta)

deaths_by_country = pd.merge( deathRateData, df_meta, on='Code').sort_values(by='profile').reset_index(drop=True)
deaths_by_country['rank'] = deaths_by_country.index+1
deaths_by_country=deaths_by_country.dropna()

# st.write(deaths_by_country)

chart = alt.Chart(deaths_by_country).mark_circle(size=60).encode(
        y='rank',
        x=alt.X('profile', axis=alt.Axis(title='Fraction of Deaths Associated with Wealth (%)')),
        color='Income group',
        tooltip=['profile', 'rank', 'Income group', 'Entity'],
    ).interactive()
col1, col2=st.columns(2)

with col1:
    st.altair_chart(chart, theme="streamlit", use_container_width=True)
with col2:
    st.markdown("""
    The fraction of deaths associated with diet and affluence             
    1. above 80%, the country is high income
    2. between 60% and 80%, the country is upper middle income
    3. between 20% and 60%, the country is lower middle income
    4. below 20%, the country is low income
 """)

st.divider()

st.markdown(""" ###  Change of Economic Status """)

country_list=list(deathRateData.Entity.unique())


col1, col2=st.columns(2)

with col1:
    selected_reasons = st.multiselect('Country', country_list,[country_list[89],country_list[69]])
    st.markdown(f"There is a signficant change in the causes of death suffered , with the country's mortality closely correlated with the country's economic progression from a lower middle-income to an upper middle-income status. This shift is a testament to the intricate interplay between economic development and public health outcomes")
with col2:
    line_plots=[]   
    for country in selected_reasons:
        df=load_data_all("parsed_1")
        df_selected = df[ df.Entity==country].copy().reset_index(drop=True)
        df_selected['profile'] = df_selected.apply( profile, axis=1)
        line=alt.Chart(df_selected).mark_line(point=True).encode(
            x='Year:N',
            y=alt.Y('profile', axis=alt.Axis(title='Fraction of Deaths Associated with Wealth (%)')),
            color='Entity:N')
        line_plots.append(line)

    if(len(line_plots)>0):
        final=line_plots[0]
        for i in range(1,len(line_plots)):
           final=final+line_plots[i]
        st.altair_chart(final,use_container_width=True)
