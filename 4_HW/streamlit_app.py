import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import altair as alt


deathRateData = pd.read_csv('https://raw.githubusercontent.com/rohanramesh38/CMSE-830/main/4_HW/death_rate_and_causes.csv')

toDrop = ['G20',
    'African Region (WHO)',
    'East Asia & Pacific (WB)',
    'Eastern Mediterranean Region (WHO)',
    'Europe & Central Asia (WB)',
    'European Region (WHO)',
    'Latin America & Caribbean (WB)',
    'Middle East & North Africa (WB)',
    'North America (WB)', 'OECD Countries',
    'Region of the Americas (WHO)', 
    'South Asia (WB)', 
    'South-East Asia Region (WHO)',
    'Sub-Saharan Africa (WB)',
    'Western Pacific Region (WHO)',
    'World',
    'World Bank High Income',
    'World Bank Low Income',
    'World Bank Lower Middle Income',
    'World Bank Upper Middle Income']

deathRateData = deathRateData[~deathRateData['Entity'].isin(toDrop)]


st.title('Death rate EDA')

st.markdown("""
This app performs EDA on the death rate and causes data!
* **Data source:** [kaggle-reference](https://www.kaggle.com/datasets/bilalwaseer/death-rate-of-countries-and-its-causes/data).
""")
st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1990,2020))))

selected_country = st.sidebar.selectbox('Country', list(deathRateData.Entity.unique())[1:])

selected_reasons = st.sidebar.selectbox('Reason', list(deathRateData.columns)[3:])



def load_data(year,selected_country):
    df_country=deathRateData[deathRateData['Entity']==selected_country ]
    df_countr_and_year =df_country[df_country['Year']==year]
    return df_countr_and_year

def load_data_with_year(year):
    df_year =deathRateData[deathRateData['Year']==year]
    return df_year

selcted_data = load_data(selected_year,selected_country)

st.markdown("""
            Top 10 countries with the highest reason for death :
            """ + selected_reasons)
            
tmp = deathRateData.groupby('Entity')[selected_reasons].mean().reset_index()
tmp = tmp.sort_values(by=selected_reasons, ascending=True)
fig, ax = plt.subplots()
ax =sns.barplot(y='Entity', x=selected_reasons, data=tmp.head(10))
st.pyplot(fig)


year_df = load_data_with_year(selected_year)
year_df

fig, ax = plt.subplots()

chart=alt.Chart(year_df).mark_circle().encode(
    alt.X(selected_reasons).scale(zero=False),
    alt.Y('Year').scale(zero=False, padding=1),
    color='Entity',
)
st.altair_chart(chart)
ax=sns.scatterplot(data=year_df, y=selected_reasons,x='Year',hue='Entity')
st.pyplot(fig)





# fig, ax = plt.subplots()
# ax = sns.histplot(selcted_data, bins=20, kde=True)
# st.pyplot(fig)

# fig, ax = plt.subplots()
# correlation_matrix = deathRateData.corr()
# plot=sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# st.pyplot(fig)