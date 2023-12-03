
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


st.set_page_config(layout="wide")
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


def fill_data_with_median():
    return selcted_data.fillna(selcted_data.median(numeric_only = True))

Country_list=list(deatRateData['Country'].unique())

selected_country = st.selectbox('Country', Country_list)   

selcted_data = data_with_filtered(deatRateData,[['Country',selected_country]])

selcted_data = selcted_data.sort_values(by='Year', ascending=True)


train_size = 0.8  # Set the percentage of data for training

split_index = int(len(selcted_data) * train_size)

encoder = LabelEncoder()
selcted_data = fill_data_with_median()

for column in ["Country Code","ISO3", "Status","Entity"]:
    selcted_data[column] = encoder.fit_transform(selcted_data[column])
    selcted_data[column] = encoder.fit_transform(selcted_data[column])




train=selcted_data[['Alochol use', 'Unsafe water source', 'Air pollution', 'Low bone mineral density', 'Respitatory Mortality','Cardio vascular Mortality','POP','Unsafe Sanitation Mortality']]
test=["Life Expectancy"]




X_train ,x_test= train[split_index:], train[:split_index]
Y_train, y_test = test[split_index:], test[:split_index]






scalar_radio = st.radio("Which Scalar to use",["MinMaxScaler", "StandardScaler"])

my_scalar = StandardScaler()

if(scalar_radio=="MinMaxScaler"):
    my_scalar=MinMaxScaler()



X_train= my_scalar.fit_transform(X_train)
x_test= my_scalar.transform(x_test)

model_list = pd.DataFrame(columns=['Model', 'Training Score', 'Test R2 Score'])



def select_model(model_name):
    global model_list  
    
    model = model_name
    
    model.fit(X_train, Y_train)
    
    train_score = model.score(X_train, Y_train)
    

    predictions = np.round(model.predict(x_test), decimals = 1)
    
    test_r2_score = r2_score(y_test, predictions)
    
    model_scores = pd.DataFrame({'Model': [model_name], 'Training Score': [train_score], 'Test R2 Score': [test_r2_score]})
    
    model_list = pd.concat([model_list, model_scores], ignore_index = True)




if('LinearRegression' in selected_models):
    select_model(LinearRegression())

if('SVR' in selected_models):
    select_model(SVR(C = 9.0, epsilon = 0.9, kernel = 'rbf'))


if('RandomForestRegressor' in selected_models):
    n_estimator = st.slider('n_estimator',min_value=50, max_value=300, value=100, step=25 )
    max_depth = st.slider('max_depth',min_value=1, max_value=20, value=7, step=1 )
    min_samples_split=st.slider('min_samples_split',min_value=1, max_value=20, value=5, step=1 )
    select_model(RandomForestRegressor(n_estimator,max_depth,min_samples_split))


if('GradientBoostingRegressor' in selected_models):
    n_estimator = st.slider('n_estimator',min_value=50, max_value=300, value=100, step=25 )
    max_depth = st.slider('max_depth',min_value=1, max_value=20, value=7, step=1 )
    min_samples_split=st.slider('min_samples_split',min_value=1, max_value=20, value=5, step=1 )
    select_model(GradientBoostingRegressor(n_estimator,max_depth,min_samples_split))

for i in range(4):
    model_list.rename(index = {i : models[i]}, inplace = True)
    
model_list.drop(columns= "Model", inplace = True)

#model_list.plot(kind = "bar", figsize = (15,6), width = 0.6)


# alt.Chart(model_list).mark_bar().encode(
#     x='year:O',
#     y='sum(yield):Q',
#     color='year:N',
#     column='site:N'
# )
# plt.xticks(rotation = 0)
# plt.show()
st.write(model_list)
    



