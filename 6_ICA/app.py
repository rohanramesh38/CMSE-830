import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import altair as alt
from altair import datum
import math
import numpy as np
option = st.selectbox('Data',('Mpg', 'Dataset points'))

st.write('data selected:', option)
data=pd.DataFrame( { 'x':[ 0.5,1.0],'y':[2.0,1.0] })

def rbf_calculation(x, centers, weights, bandwidth):
    return np.sum(weights * np.exp(-((x - centers) / bandwidth)**2), axis=1)
def plot_rbf_val(centers, weights, bandwidth):
    x = np.linspace(-10, 10, 1000)
    y = rbf(x[:, np.newaxis], centers, weights, bandwidth)
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='RBF Function', color='b')
    plt.scatter(centers, weights, color='r', marker='o', label='Centers')
    plt.xlabel('X')
    plt.ylabel('RBF Value')
    plt.title('Radial Basis Function')
    plt.legend()
    plt.grid(True)
    fig=plt.show()
    st.pyplot(fig)

def linear_regression_mse(m, b, x_list, y_list):
    z = 0
    for i in range(len(x_list)):
        z += (y_list[i] - (m * x_list[i] + b))**2
    return z

def linear_regression_mae(m, b, x_list, y_list):
    z = 0
    for i in range(len(x_list)):
        z += abs(y_list[i] - (m * x_list[i] + b))
    return z

def rbf_function(x, center, width):
    return np.exp(-0.5 * ((x - center) / width) ** 2)

def rbf(x, centers, weights, bandwidth):
    return np.sum(weights * np.exp(-((x - centers) / bandwidth)**2), axis=1)

def plot_rbf(centers, weights, bandwidth):
    x = np.linspace(-10, 10, 1000)  # X values for plotting
    y = rbf(x[:, np.newaxis], centers, weights, bandwidth)  # RBF values for given X
    plt.figure(figsize=(8, 6))
    fig=plt.plot(x, y, label='RBF Function', color='b')
    plt.scatter(centers, weights, color='r', marker='o', label='Centers')
    plt.grid(True)
    fig=plt.show()



def get_points(slope,intercept,x):
    lst=[]
    xvaluesline=[]
    yvaluesline=[]


    for xvalue in range(math.floor(min(x)),math.floor(max(x)),1):
        yvalue=slope*xvalue+intercept
        xvaluesline.append(xvalue)
        yvaluesline.append(yvalue)
    
    dfpoints=pd.DataFrame({
        'xvaluesline':xvaluesline,
        'yvaluesline':yvaluesline
    })
    return dfpoints

if(option=='Mpg'):
    data=sns.load_dataset("mpg")
    data=data.drop(columns = ["cylinders", "model_year", "origin", "name"])
    st.write(data)


else:
    st.write(data)

coulmns=list(data.columns)


options = st.multiselect("coulmns",coulmns,[coulmns[0], coulmns[1]])
if(len(options)!=2):
    st.write("select x and y cordinates")
else:
    st.write("x = ",options[0])
    st.write("y = ",options[1])

    xname=options[0]
    yname=options[1]


    points=data[[xname,yname]]

    chart=alt.Chart(points).mark_point().encode(
        x=xname,
        y=yname,
    )

    slope = st.slider('Select value for m', -10.0, 10.0,0.05)

    intercept = st.slider('Select value for b', 2*min(data[yname]), 2*max(data[yname]),1.0)

    new_points=get_points(slope,intercept,data[xname])
    st.write(new_points)


    st.write("linear_regression_mse ",linear_regression_mse(slope,intercept,data[xname],data[yname]))
    st.write("linear_regression_mae ",linear_regression_mae(slope,intercept,data[xname],data[yname]))
    

    chart2=alt.Chart(new_points).mark_line().encode(
    x="xvaluesline:Q",
    y="yvaluesline:Q",)
    fchart=chart+chart2
    st.altair_chart(fchart)

    cntrs = np.array([-5, 1, 6])
    wghts = np.array([0.1, 0.3, 0.6])
    band = st.slider('bandwidth', 1, 10,1)

    plot_rbf_val(cntrs, wghts, band)
 
