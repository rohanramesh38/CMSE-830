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
st.set_page_config(layout="wide")
deathRateData=load_data_all()



st.title('Reason Based Analysis')

st.markdown(
    """
    The dataset is of various nations of the world. We neglect the immediate cause of death and instead focus on long term causes due to various factors.
    """,True)



col1, col2,col3 = st.columns(3)

df1={
"Category": ["affluence","","","","vices","","",""],
"Reason":["high blood pressure","diabetes","being overweight","low physical activity","alcohol","second hand smoke","smoking","drugs"]
}
df1=pd.DataFrame(df1)

df2={
"Category": ["environmental","","","","childhood","","","","misc"],
"Reason":["air pollution - various types","unsafe water source","unsafe sanitation","lno access to handwashing facility","low birthweight","child wasting","child stunting","discontinued or non-exclusive breast feeding","unsafe sex"]
}
df2=pd.DataFrame(df2)

df3={
"Category": ["poor diet","","","","poor nutition","","",""],
"Reason":["high in sodium","low in whole grains","low in nuts and seeds","low in fruits","low in vegetables","vitamin A deficiency","iron deficiency","low bone mineral density"]
}
df3=pd.DataFrame(df3)

with col1:
   st.dataframe(df1,hide_index=True,use_container_width=True)
with col2:
   st.dataframe(df2,hide_index=True,use_container_width=True)
with col3:
   st.dataframe(df3,hide_index=True,use_container_width=True)
st.markdown("""

Certainly, while it's true that individuals across various income groups experienced 
            some level of impact from excess weight (which labeled as a concern related to prosperity),
             the significance of this issue increased with higher income levels. Broadly speaking, factors 
            categorized as affluence, vices, and poor dietary habits appeared to become more prominent as 
            income levels rose, while factors categorized as environmental, childhood-related, inadequate
             nutrition, and miscellaneous factors appeared to gain relevance as income levels declined.
""")

st.divider()


col1, col2 = st.columns(2)

reason_list=list(deathRateData.columns)[3:-3]
country_list=list(deathRateData.Entity.unique())

#selected_country = st.sidebar.selectbox('Country', list(deathRateData.Entity.unique())[1:])
#selected_country = st.sidebar.selectbox('Country', list(deathRateData.Entity.unique())[1:])

with col1:
   selected_reasons = st.selectbox('Reason', reason_list)
   st.write("")

   #selected_continent = st.selectbox('Continent', list(deathRateData.Continent.unique()))   
   #st.write("")

with col2:
   #selected_year = st.selectbox('Year', list(reversed(range(1990,2020))))
   #st.write("")
   selected_continent = st.selectbox('Continent', list(deathRateData.Continent.unique())+['All'] )
  


heading="Top 10 countries in "+ selected_continent +" with high " + selected_reasons
st.header(heading)


selected_year = st.slider('Year', 1990,2020)

selcted_data = data_with_filtered(deathRateData,[['Continent',selected_continent],['Year',selected_year]])
selcted_data = selcted_data.sort_values(by=selected_reasons, ascending=False)
tab1, tab2= st.tabs(["Map", "Bar"])

with tab1:
   map_fig=choropleth(data_frame=selcted_data, locations='ISO3', color=selected_reasons, hover_name='Entity')
   map_fig.update_geos(fitbounds="locations", visible=False)
   map_fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
   st.plotly_chart(map_fig,use_container_width=True)
   st.markdown( "The intensity shade indicates the amount contributed by the Reason ")
with tab2:
   tmp = selcted_data.groupby('Entity')[selected_reasons].mean().reset_index()
   tmp = tmp.sort_values(by=selected_reasons, ascending=False)

   chart = alt.Chart(tmp.head(10)).mark_bar().encode(
        x=selected_reasons,
        y=alt.Y('Entity:N').sort('-x')
    )
   st.altair_chart(chart, theme="streamlit", use_container_width=True)

   st.markdown( "We can infer from the Graph that the Country with Highiest "+selected_reasons +" "+tmp.head(10)['Entity'].iloc[0])
   
   
   
#-------------------------------------------
st.divider()

reas= st.selectbox('Reason ', reason_list)
st.write("")
st.subheader("Yearly trend for "+reas)
selection = alt.selection_multi(fields=['series'], bind='legend')


col1, col2 = st.columns(2)

fig2 = px.area(deathRateData, x="Year", y=reas, color="Continent", line_group="Entity")
with col1:
   st.plotly_chart(fig2, theme="streamlit")
   st.markdown("Yearly Relation for the reason is plotted denoting the trend of specific reason")

with col2:
   country=""
   
   country = st.selectbox('Country', list(deathRateData.Entity.unique())[1:])
   st.markdown(""" #### Trend for Country """ + country)
   
   f_data=data_with_filtered(deathRateData,[['Entity',country]])
   fig = px.pie(f_data, values=reas, names='Year')
   st.plotly_chart(fig, theme="streamlit")

   st.markdown("The width of the pie of the chart denotes the change in the reason at a specific year")

st.divider()
#--------------------------------------------

selected_country = st.selectbox('Country', country_list)


st.subheader("Top 5 Reason for mortality of "+selected_country)

st.markdown ("""

Knowing the leading causes of death helps in planning and allocating resources for healthcare infrastructure, 
             including hospitals, clinics, and specialized facilities. 
             It enables the optimization of medical services to address specific health challenges.
             Governments can formulate healthcare policies and regulations based on the leading causes of death. 
             These policies can focus on improving healthcare delivery, enhancing access to care, and implementing measures to reduce the prevalence of specific diseases.
""")

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

st.plotly_chart(fig,theme="streamlit",use_container_width=True)


st.divider()


#st.write(df_plot)
#---------------------------------------------

st.subheader("Mortality rate based on the population growth over the years ")

col_1, col_2 = st.columns(2)

#selected_country = st.sidebar.selectbox('Country', list(deathRateData.Entity.unique())[1:])
#selected_country = st.sidebar.selectbox('Country', list(deathRateData.Entity.unique())[1:])

with col_1:
   r1 = st.selectbox('Reason 1 ', reason_list)
   st.write("")


with col_2:
   selected_country = st.multiselect('Country', country_list,[country_list[69],country_list[149],country_list[86]])
   st.write("")

year='All'

year_data = data_with_filtered(deathRateData,[['Year',year],['Entity',selected_country]])
year_data[r1]=year_data[r1]/year_data["POP"]

#st.write(year_data)
year_data=year_data.dropna()

fig = px.scatter(year_data, y=r1, x="Year",size="POP", color="Continent",hover_name="Entity", log_x=True, size_max=60)
st.plotly_chart(fig, theme="streamlit",use_container_width=True)

st.markdown("""

1. **Decreasing Mortality Rates**: In many developed countries, as healthcare access and medical advancements improve, mortality rates tend to decrease over time. This pattern is often observed as a population grows and ages, thanks to better disease prevention, early diagnosis, and improved treatment options.
2. **Stable Mortality Rates**: Some regions may experience relatively stable mortality rates, especially if healthcare infrastructure and public health measures remain consistent. Even with population growth, the overall health and mortality patterns remain relatively unchanged.
3. **Fluctuating Mortality Rates**: In certain situations, mortality rates may fluctuate due to periodic outbreaks of infectious diseases, changes in lifestyle habits, or economic shifts. These fluctuations can occur within a growing or stable population.
""")
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
