
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rf = pickle.load(open('rf_model.pkl','rb'))
lr = pickle.load(open('lr_model.pkl','rb'))
knn = pickle.load(open('knn_model.pkl','rb'))

st.title('Iris Web App')

ml_model = ['Logistic Regression','RandomForest Classiifer','KNN Classifier']

option = st.sidebar.selectbox('Select the ML model',ml_model)

sl = st.slider('Select Sepal Length', 0.0,10.0)
sw = st.slider('Select Sepal Width', 0.0,10.0)
pl = st.slider('Select Petal Length', 0.0,10.0)
pw = st.slider('Select Petal Width', 0.0,10.0)

test =  [[sl,sw,pl,pw]]

st.write('Test Data')
st.write(test)


if st.button('Classify'):
    res = None
    if option=='Logistic Regression':
        st.success(lr.predict(test)[0])
    elif option=="RandomForest Classiifer":
        st.success(rf.predict(test)[0])
    else:
        st.success(knn.predict(test)[0])

# st.success(res)


