
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
from sklearn.neighbors import KNeighborsClassifier
# import libraries
from sklearn.metrics import ConfusionMatrixDisplay


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from helper.data_helper import load_data_all,data_with_filtered
st.set_page_config(layout="wide")
deatRateData =load_data_all()



st.subheader("Life Expectancy Prediction")

st.image("https://odh.ohio.gov/wps/wcm/connect/gov/463db345-90eb-4636-a9ba-c4e9e63937ca/GettyImages-831551460_1280x480.jpg?MOD=AJPERES&CACHEID=ROOTWORKSPACE.Z18_K9I401S01H7F40QBNJU3SO1F56-463db345-90eb-4636-a9ba-c4e9e63937ca-o9F8OTq")

data_on = st.toggle('Show data')

if(data_on):
    st.write(deatRateData.head())



st.subheader("Regression")


def fill_data_with_median():
    return selcted_data.fillna(selcted_data.median(numeric_only = True))

Country_list=list(deatRateData['Country'].unique())

selected_country = st.selectbox('Country', Country_list)   

selcted_data = data_with_filtered(deatRateData,[['Country',selected_country]])

#selcted_data = selcted_data.sort_values(by='Year', ascending=True)
selcted_data = selcted_data.sample(frac=1, random_state=42)  # frac=1 indicates the entire DataFrame


train_size = 0.8  # Set the percentage of data for training

split_index = int(len(selcted_data) * train_size)
scalar_radio = st.radio("Which Scalar to use",["MinMaxScaler", "StandardScaler"])

my_scalar = StandardScaler()

if(scalar_radio=="MinMaxScaler"):
    my_scalar=MinMaxScaler()

models = ['LinearRegression','GradientBoostingRegressor' ,  'RandomForestRegressor','SVR']

cols = st.columns(len(models))
checks=[]

for i in range(0,len(models)):
    with cols[i]:
        box=st.checkbox(models[i],value=True)
        checks.append(box)

selected_models = [value for value, boolean in zip(models, checks) if boolean]


encoder = LabelEncoder()
selcted_data = fill_data_with_median()

for column in ["Country Code","ISO3", "Status"]:
    selcted_data[column] = encoder.fit_transform(selcted_data[column])
    selcted_data[column] = encoder.fit_transform(selcted_data[column])





for val in ['Alochol use', 'Unsafe water source', 'Air pollution', 'Low bone mineral density', 'Respitatory Mortality','Cardio vascular Mortality','Poputaltion','Unsafe Sanitation Mortality']:
    selcted_data[val]=selcted_data[val]/selcted_data["Poputaltion"]

train=selcted_data[[ 'Air pollution',  'Respitatory Mortality','Cardio vascular Mortality','Unsafe Sanitation Mortality','Low bone mineral density','Year']]

test=selcted_data["Life Expectancy"]
train=my_scalar.fit_transform(train)

validation =selcted_data[selcted_data['Year'].isin(list(range(2000,2021))) ].copy()

validation=validation.sort_values(by='Year', ascending=True)

validation_x=validation[[ 'Air pollution',  'Respitatory Mortality','Cardio vascular Mortality','Unsafe Sanitation Mortality','Low bone mineral density','Year']]
validation_x=my_scalar.transform(validation_x)
validation_y=validation["Life Expectancy"]






selcted_data=selcted_data.sort_values(by='Year', ascending=True)





X_train ,x_test= train[:split_index], train[split_index:]
Y_train, y_test = test[:split_index], test[split_index:]



model_list = pd.DataFrame(columns=['Model', 'Training Score', 'Test R2 Score'])

model_preds = pd.DataFrame(columns=['Model','Preds','Year'])

chart_full_x=sorted(list(selcted_data['Year'].unique()))
chart_full_y=test
chart_validation_x=list(range(2000,2020))



def select_model(model_name):
    global model_list  
    
    model = model_name
    model.fit(X_train, Y_train)
    
    train_score = model.score(X_train, Y_train)
    
    predictions = np.round(model.predict(x_test), decimals = 1)
    
    test_r2_score = r2_score(y_test, predictions)

    chart_validation_y=model.predict(validation_x)
    
    model_scores = pd.DataFrame({'Model': [model_name], 'Training Score': [train_score], 'Test R2 Score': [test_r2_score]})
    
    model_list = pd.concat([model_list, model_scores], ignore_index = True)
    # st.write(train)
    # st.write(chart_full_y)
    # st.write(chart_full_x)

    # st.write(validation_x)
    # st.write(chart_validation_y)
    

    data = pd.DataFrame({
    'years': chart_full_x,
    'Life Expectancy': chart_full_y,
    })
    data2=pd.DataFrame({
    'years':chart_validation_x,
    'Life Expectancy':chart_validation_y
    })

    # Create Altair line chart
    line_chart = alt.Chart(data).mark_line().encode(
        x='years',
        y='Life Expectancy',
    ).properties(
        width=600,
        height=300,
        title=str(model_name)
    )

    # Create a second line (trace) on the same chart
    trace_line = alt.Chart(data2).mark_line(color='red').encode(
        x='years',
        y='Life Expectancy',
        
    )

    # Combine both charts
    combined_chart = (line_chart + trace_line)


    st.altair_chart(combined_chart)




    

def change(show):
    if(show):
        st.write(model_list)





col_1,col_2=st.columns(2)
if('LinearRegression' in selected_models):
    
    with col_1:
        select_model(LinearRegression())



if('GradientBoostingRegressor' in selected_models):
    with col_2:
        n_estimator_2 = st.slider('n_estimators',min_value=50, max_value=300, value=100, step=25 )
        max_depth_2 = st.slider('max_depth ',min_value=1, max_value=20, value=7, step=1 )
        min_samples_split_2=st.slider('min_samples_split ',min_value=1, max_value=20, value=5, step=1 )
        select_model(GradientBoostingRegressor(n_estimators=n_estimator_2,max_depth=max_depth_2,min_samples_split=min_samples_split_2,random_state=42))


col1,col2=st.columns(2)

if('RandomForestRegressor' in selected_models):
    with col1:
        n_estimator = st.slider('n_estimator',min_value=50, max_value=300, value=100, step=25 )
        max_depth = st.slider('max_depth',min_value=1, max_value=20, value=7, step=1 )
        min_samples_split=st.slider('min_samples_split',min_value=1, max_value=20, value=5, step=1 )
        select_model(RandomForestRegressor(n_estimators=n_estimator,max_depth=max_depth,min_samples_split=min_samples_split,random_state=42))


if('SVR' in selected_models):
    with col2:
        kernel= st.selectbox('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        degree=  st.slider('degree',min_value=2, max_value=10, value=3, step=1 )
        select_model(SVR(C = 9.0, epsilon = 0.9, kernel = kernel,degree=degree))



for i in range(4):
    model_list.rename(index = {i : models[i]}, inplace = True)
    
model_list.drop(columns= "Model", inplace = True)

#model_list.plot(kind = "bar", figsize = (15,6), width = 0.6)
fig=model_list.plot(kind = "bar", figsize = (10,2), width = 0.5)


st.pyplot(fig.figure)
on = st.toggle('Show scores')
change(on)




st.subheader("Classification")

from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, :2]  # Using only the first two features for visualization
y = iris.target


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Support Vector Machine classifier
svm_classifier = SVC(kernel='linear')

# Train the classifier using the training data
svm_classifier.fit(X_train, y_train)

# Make predictions on test data
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)

# Visualize decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# # Streamlit app
# st.title('SVC Classification on Iris Dataset')

# # Scatter plot
# st.write('Scatter Plot of Sepal Length vs Sepal Width')
# plt.figure(figsize=(8, 6))
# plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
# plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='viridis')
# plt.xlabel('Sepal Length')
# plt.ylabel('Sepal Width')
# plt.title('Decision Boundaries')
# st.pyplot(plt)



# alt.Chart(model_list).mark_bar().encode(
#     x='year:O',
#     y='sum(yield):Q',
#     color='year:N',
#     column='site:N'
# )
# plt.xticks(rotation = 0)
# plt.show()

selcted_data = data_with_filtered(deatRateData,[['Country','All']])
selcted_data = selcted_data.sample(frac=1, random_state=42)  # frac=1 indicates the entire DataFrame
encoder = LabelEncoder()
selcted_data = fill_data_with_median()

for column in ["Country Code","ISO3", "Status",]:
    selcted_data[column] = encoder.fit_transform(selcted_data[column])
    selcted_data[column] = encoder.fit_transform(selcted_data[column])

for val in ['Alochol use', 'Unsafe water source', 'Air pollution', 'Low bone mineral density', 'Respitatory Mortality','Cardio vascular Mortality','Nutrition Defeciency Mortality','Unsafe Sanitation Mortality']:
    selcted_data[val]=selcted_data[val]/selcted_data["Poputaltion"]


selcted_data = selcted_data.groupby('Entity').agg({'Alochol use': 'sum', 'Air pollution': 'sum','Value': 'sum' ,'Status': 'max','Life Expectancy':'sum','Cardio vascular Mortality':'sum','Unsafe Sanitation Mortality':'sum'})

Y_values=selcted_data["Status"].to_numpy()


X_values=selcted_data[["Life Expectancy","Air pollution"]].to_numpy()



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_values, Y_values, test_size=0.2, random_state=42)

# Initialize the Support Vector Machine classifier
svm_classifier = SVC(kernel='linear')
clf_knn = KNeighborsClassifier(n_neighbors=2)

# Train the classifier using the training data
svm_classifier.fit(X_train, y_train)
clf_knn.fit(X_train, y_train)



# Make predictions on test data
y_pred = svm_classifier.predict(X_test)
y_pred_knn = clf_knn.predict(X_test)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
accuracy_knn = np.mean(y_pred == y_pred_knn)

# Visualize decision boundaries
x_min, x_max = X_values[:, 0].min() - 1, X_values[:, 0].max() + 1
y_min, y_max = X_values[:, 1].min() - 1, X_values[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Streamlit app
st.subheader(' Classification on Developing Developed Country')


tab1,tab2=st.tabs(['SVC','KNN'])



with tab1:
    x_min, x_max = X_values[:, 0].min() - 1, X_values[:, 0].max() + 1
    y_min, y_max = X_values[:, 1].min() - 1, X_values[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Scatter plot
    fig=plt.figure(figsize=(8, 3))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
    plt.scatter(X_values[:, 0], X_values[:, 1], c=Y_values, edgecolor='k', cmap='viridis')
    plt.xlabel('Life Expectancy')
    plt.ylabel('Adult Mortality')
    plt.title('Decision Boundaries')
    st.pyplot(fig)
    titles_options = [
        ("Normalized confusion matrix", "true"),
    ]

    class_names=["Developing", "Developed"]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            svm_classifier,
            X_test,
            y_test,
            display_labels=class_names,
            cmap=plt.cm.Blues,
            normalize=normalize,
        )
        disp.ax_.set_title(title)
        
    confusion = st.toggle('Show confusion Matrix')

    if(confusion):
        st.pyplot(plt)
with tab2:
    x_min, x_max = X_values[:, 0].min() - 1, X_values[:, 0].max() + 1
    y_min, y_max = X_values[:, 1].min() - 1, X_values[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = clf_knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Scatter plot
    fig=plt.figure(figsize=(8, 3))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='plasma')
    plt.scatter(X_values[:, 0], X_values[:, 1], c=Y_values, edgecolor='k', cmap='plasma')
    plt.xlabel('Life Expectancy')
    plt.ylabel('Adult Mortality')
    plt.title('Decision Boundaries')
    st.pyplot(fig)


    titles_options = [
        ("Normalized confusion matrix", "true"),
    ]

    class_names=["Developing", "Developed"]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            clf_knn,
            X_test,
            y_test,
            display_labels=class_names,
            cmap=plt.cm.Blues,
            normalize=normalize,
        )
        disp.ax_.set_title(title)
        
    confusion = st.toggle('Show confusion Matrix ')

    if(confusion):
        st.pyplot(plt)



