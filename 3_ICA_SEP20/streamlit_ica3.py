import streamlit as st
import seaborn as sns
import pandas as pd
import altair as alt

df=pd.read_csv('https://raw.githubusercontent.com/rohanramesh38/CMSE-830/main/3_ICA_SEP20/data.csv')

df_area=df[['area_se','area_mean','area_worst','diagnosis']]
df_area=df_area.replace({'diagnosis':{'M':0,'B':1}})
df_area

plot=sns.pairplot(df_area,hue='diagnosis')
st.pyplot(plot.fig)


option = st.selectbox('select column name for showing the displot',('area_worst', 'area_se','area_mean' ))
st.write('You selected:', option)
plt2=sns.displot(df_area, x=option, hue="diagnosis", multiple="stack")
st.pyplot(plt2.fig)

#alt_handle = alt.Chart(df_area).mark_circle(size=60).encode(x='area_mean', y=option,color='diagnosis', tooltip=['ash', 'magnesium','proanthocyanins']).interactive()
#st.altair_chart(alt_handle)

working_data = df[['diagnosis', 'radius_mean', 'radius_se', 'radius_worst',
                     'symmetry_mean', 'symmetry_se', 'symmetry_worst']]

# looking at the radius data
radius_data = df[['diagnosis', 'radius_mean', 'radius_se', 'radius_worst']]
sns.pairplot(radius_data, hue = "diagnosis")

# symmetry data comparison
symmetry_data = df[['diagnosis', 'symmetry_mean', 'symmetry_se', 'symmetry_worst']]
sym_plot = sns.pairplot(symmetry_data, hue = "diagnosis")

mean_radius_plot = sns.violinplot(data= radius_data, x="radius_mean", y="diagnosis")

worse_radius_plot = sns.violinplot(data= radius_data, x="radius_worst", y="diagnosis")

rad_sym_heatmap = sns.heatmap(working_data.corr(), annot=True)

st.pyplot(mean_radius_plot.get_figure())

st.pyplot(worse_radius_plot.get_figure())

st.pyplot(rad_sym_heatmap.get_figure())

cancer = df[["diagnosis", "perimeter_mean", "perimeter_se", "perimeter_worst"]]
cancer
cancer.groupby("diagnosis", group_keys=True).apply(lambda x: x)

# per_violin_plot = sns.violinplot(x='perimeter_mean', y='diagnosis', data=cancer, hue='diagnosis', split=True)

# st.pyplot(per_violin_plot.get_figure())


s_df=df[['id', 'diagnosis','smoothness_mean','smoothness_se', 'smoothness_worst']]


plot=sns.pairplot(s_df,hue='diagnosis')
st.pyplot(plot.fig)


option = st.selectbox('select a column for displot',('smoothness_mean','smoothness_se', 'smoothness_worst' ))
st.write('You selected:', option)
plt2=sns.displot(s_df, x=option, hue="diagnosis", multiple="stack")
st.pyplot(plt2.fig)
