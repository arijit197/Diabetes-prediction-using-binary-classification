from finalModel import *
import streamlit as st
import numpy as np 
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt
from plotly import graph_objs as go




st.title("Diabetes Predictor")
st.image("diabetes.jpeg",width = 800)
nav = st.sidebar.radio("Navigation",["Home","Prediction","Contribute"])
if nav == "Home":
    
    st.subheader("Dataset")
    if st.checkbox("Show Table"):
        st.table(Imputed_Dataset)
    
    st.subheader("Scatterplot")

    selected_variable = st.selectbox("Select a variable", ["Glucose & BloodPressure","BMI & SkinThickness","Glucose & Insulin","BMI & Glucose"])

    if selected_variable == "Glucose & BloodPressure":

        scatterplot = sns.scatterplot(data=Imputed_Dataset, x="Glucose", y="BloodPressure", hue="Outcome")
        st.pyplot(scatterplot.figure)

    if selected_variable == "Glucose & Insulin":

        scatterplot = sns.scatterplot(data=Imputed_Dataset, x="Glucose", y="Insulin", hue="Outcome")
        st.pyplot(scatterplot.figure)

    if selected_variable == "BMI & SkinThickness":

        scatterplot = sns.scatterplot(data=Imputed_Dataset, x="BMI", y="SkinThickness", hue="Outcome")
        st.pyplot(scatterplot.figure)

    if selected_variable == "BMI & Glucose":

        scatterplot = sns.scatterplot(data=Imputed_Dataset, x="BMI", y="Glucose", hue="Outcome")
        st.pyplot(scatterplot.figure)


    st.subheader("Correlation figure")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(Imputed_Dataset.corr())
    st.pyplot(fig)


    st.subheader("Accuracy score of this model : ")
    st.write(str(round(acc,4)*100)+"%")

            
    
if nav == "Prediction":
    st.header("Check wheather you have Diabetes or not")

    Glu = st.slider("Enter your Glucose level",min_value = 40,max_value = 200)
    Bp = st.number_input("Enter your BloodPressure",min_value = 0,max_value = 130)
    St = st.number_input("Enter your SkinThickness",min_value = 0,max_value = 100)
    Ins = st.slider("Enter your Insulin level",min_value = 0,max_value = 600)
    Bmi = st.number_input("Enter your BMI",min_value = 0.00,max_value = 70.00)
    Age = st.slider("Enter your Age",min_value = 18,max_value = 100)
    
    val = np.array([Glu,Bp,St,Ins,Bmi,Age]).reshape(1,-1)
    pred = model.predict(val)

    if st.button("Predict"):
        if pred != 0:
            st.warning(f"You might have diabetes. It is advisable to consult a doctor for further evaluation.")
        else:
            st.success(f"You Don't have Diabetes")


if nav == "Contribute":
    st.header("Contribute to our dataset")

    Glu = st.number_input("Enter your Glucose level",min_value = 0,max_value = 300)
    Bp = st.number_input("Enter your BloodPressure",min_value = 0,max_value = 300)
    St = st.number_input("Enter your SkinThickness",min_value = 0,max_value = 200)
    Ins = st.number_input("Enter your Insulin level",min_value = 0,max_value = 800)
    Bmi = st.number_input("Enter your BMI",min_value = 0.00,max_value = 200.00)
    Age = st.number_input("Enter your Age",min_value = 0,max_value = 200)

    st.subheader("Did you have Diabetes?")
    
    op = st.radio("Your answer",["Yes","No"])
    if op == "Yes" : op = 1
    else: op = 0
    
    if st.button("submit"):
        to_add = {"Glucose":[Glu],"BloodPressure":[Bp],"SkinThickness":[St],"Insulin":[Ins],"BMI":[Bmi],"Age":[Age],"Outcome":[op]}
        to_add = pd.DataFrame(to_add)
        to_add.to_csv("updatedDataset.csv",mode='a',header = False,index= False)
        st.success("Submitted. Thank you so much for the Contribution")
    