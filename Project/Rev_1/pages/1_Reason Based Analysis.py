import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import altair as alt
from countryinfo import CountryInfo
from plotly.express import choropleth
from helper.data_helper import load_data_all,data_with_filtered
import plotly.express as px
import numpy as np

deathRateData=load_data_all()


col1, col2 = st.columns(2)

reason_list=list(deathRateData.columns)[3:-3]
country_list=list(deathRateData.Entity.unique())

#selected_country = st.sidebar.selectbox('Country', list(deathRateData.Entity.unique())[1:])
#selected_country = st.sidebar.selectbox('Country', list(deathRateData.Entity.unique())[1:])

with col1:
   selected_reasons = st.selectbox('Reason', reason_list)
   st.write("")
   format = st.selectbox('View', ['Map','Bar'])
   #selected_continent = st.selectbox('Continent', list(deathRateData.Continent.unique()))   
   #st.write("")

with col2:
   #selected_year = st.selectbox('Year', list(reversed(range(1990,2020))))
   #st.write("")
   selected_continent = st.selectbox('Continent', list(deathRateData.Continent.unique())+['All'] )
  


heading="Top 10 countries in "+ selected_continent +" with high " + selected_reasons
st.header(heading)
selcted_data = data_with_filtered(deathRateData,[['Continent',selected_continent]])
selcted_data = selcted_data.sort_values(by=selected_reasons, ascending=False)
if(format=='Bar'):
   tmp = selcted_data.groupby('Entity')[selected_reasons].mean().reset_index()
   tmp = tmp.sort_values(by=selected_reasons, ascending=False)
   fig, ax = plt.subplots()
   ax =sns.barplot(y='Entity', x=selected_reasons, data=tmp.head(10))
   st.pyplot(fig)
elif(format=='Map'):
   map_fig=choropleth(data_frame=selcted_data, locations='ISO3', color=selected_reasons, hover_name='Entity')
   map_fig.update_geos(fitbounds="locations", visible=False)
   map_fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
   st.plotly_chart(map_fig,use_container_width=True)
   
#-------------------------------------------


reas= st.selectbox('Reason ', reason_list)
st.write("")
st.subheader("Yearly trend for "+reas)

fig2 = px.area(deathRateData, x="Year", y=reas, color="Continent", line_group="Entity")
st.plotly_chart(fig2, theme="streamlit")

country = st.selectbox('Country', list(deathRateData.Entity.unique())[1:])

f_data=data_with_filtered(deathRateData,[['Entity',country]])

fig = px.pie(f_data, values=reas, names='Year')
st.plotly_chart(fig, theme="streamlit")

#--------------------------------------------

selected_country = st.selectbox('Country', country_list)

st.subheader("Top 5 Reason for mortality in "+selected_country)

f_data=data_with_filtered(deathRateData,[['Entity',selected_country]],["Year","POP"])

f_data=f_data.groupby( ['Entity'] ).agg(np.sum)
first_row = f_data.iloc[0]
sorted_values = first_row.sort_values()
df_plot={"Reason":[],"Count":[]}

# Get the column names based on the sorted values
sorted_columns = list(sorted_values.index)[::-1]
sorted_columns=sorted_columns[1:6]
#st.write(sorted_columns)

for val in sorted_columns:
   df_plot['Reason'].append(val)
   df_plot['Count'].append(f_data[val].iloc[0])

fig = px.bar(df_plot, y='Count', x='Reason')

st.plotly_chart(fig,theme="streamlit")




#st.write(df_plot)
#---------------------------------------------

col_1, col_2 = st.columns(2)

#selected_country = st.sidebar.selectbox('Country', list(deathRateData.Entity.unique())[1:])
#selected_country = st.sidebar.selectbox('Country', list(deathRateData.Entity.unique())[1:])

with col_1:
   r1 = st.selectbox('Reason 1', reason_list)
   st.write("")

   year  = st.selectbox('Year', list(reversed(range(1990,2020))))
   st.write("")

with col_2:
   r2 = st.selectbox('Reason 2', reason_list)
   st.write("")

st.subheader(r1+" vs "+r2)

year_data = data_with_filtered(deathRateData,[['Year',year]])

fig = px.scatter(year_data, x=r1, y=r1,size=r1, color="Continent",hover_name="Entity", log_x=True, size_max=60)
st.plotly_chart(fig, theme="streamlit")

# year_df = data_with_filtered(deathRateData,[['Year',selected_year]])
# year_df

# fig, ax = plt.subplots()

# chart=alt.Chart(year_df).mark_circle().encode(
#     alt.X(selected_reasons).scale(zero=False),
#     alt.Y('Year').scale(zero=False, padding=1),
#     color='Entity',
# )
# st.altair_chart(chart)
# ax=sns.scatterplot(data=year_df, y=selected_reasons,x='Year',hue='Entity')
# st.pyplot(fig)
