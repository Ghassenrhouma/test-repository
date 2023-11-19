import pickle
import numpy as np
import pandas as pd
import warnings
import altair as alt
from sklearn.preprocessing import LabelEncoder
import streamlit as st
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

rfc_loaded = data["model"]
workclass = data['workclass']
education = data['education']
sexe = data['sexe']
Race = data['Race']
Age = data['Age']
Relationship = data['Relationship']
marital_status = data['marital_status']
Native_Country = data['Native_Country']
Hours_per_week = data['Hours_per_week']
Salary = data['Salary']

def show_predict_page():
    st.title("Salary Prediction")
    st.write("""### We need some information to predict the salary""")
    country = st.selectbox("Native country:", countries)
    workclass_options = [
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "Self-emp-not-inc",
        "State-gov",
        "Private",
        "Without-pay"
    ]
    workclass = st.selectbox("Work Class:", workclass_options)
    sexe_options = ["Male", "Female"]
    sexe = st.selectbox("Sexe:", sexe_options)
    martial_status_options = [
        "Married-AF-spouse",
        "Married-civ-spouse",
        "Divorced",
        "Widowed",
        "Married-spouse-absent",
        "Separated",
        "Never-married"
    ]
    martial_status = st.selectbox("Martial status:", martial_status_options)
    relationship_options = [
        "Wife",
        "Husband",
        "Not-in-family",
        "Unmarried",
        "Other-relative",
        "Own-child"
    ]
    relationship = st.selectbox("Relationship:", relationship_options)
    occupation_options = [
        "Exec-managerial",
        "Prof-specialty",
        "Protective-serv",
        "Tech-support",
        "Sales",
        "Craft-repair",
        "Transport-moving",
        "Adm-clerical",
        "Machine-op-inspct",
        "Farming-fishing",
        "Armed-Forces",
        "Handlers-cleaners",
        "Other-service",
        "Priv-house-serv"
    ]
    occupation = st.selectbox("Occupation:", occupation_options)
    education_options = [
        "HS-grad",
        "Some-college",
        "Bachelors",
        "Masters",
        "Assoc-voc",
        "11th",
        "Assoc-acdm",
        "10th",
        "7th-8th",
        "Prof-school",
        "9th",
        "12th",
        "Doctorate",
        "5th-6th",
        "1st-4th",
        "Preschool"
    ]
    education = st.selectbox("Education:", education_options)
    race_options = [
        "White",
        "Black",
        "Asian-Pac-Islander",
        "Amer-Indian-Eskimo",
        "Other"
    ]
    race = st.selectbox("Race:", race_options)
    age = st.slider("Age:", 17, 90)
    hours_per_week = st.slider("Hours per Week:", 1, 100)

    ok = st.button('Validate')
    if ok:
        input_data = pd.DataFrame({
            'NativeCountry': [country],
            'WorkClass': [workclass],
            'Sex': [sexe],
            'MaritalStatus': [martial_status],
            'Relationship': [relationship],
            'Occupation': [occupation],
            'Education': [education],
            'Race': [race],
            'Age': [age],
            'HoursPerWeek': [hours_per_week]
        })

        lbl = LabelEncoder()
        for col in input_data.columns:
            input_data[col] = lbl.fit_transform(input_data[col])

        # Make the prediction using the loaded model
        prediction = rfc_loaded.predict(input_data)
        st.write("Prediction:", prediction)

        # Display the prediction result
        if prediction[0] == 1:
            st.write("The predicted salary is greater than $50K")
        else:
            st.write("The predicted salary is less than or equal to $50K")


countries = [
    "Taiwan", "France", "Iran", "India", "Japan", "Cambodia", "Yugoslavia", "Italy", "England", "Germany",
    "Canada", "Philippines", "Hong", "China", "Greece", "Cuba", "United-States", "Hungary", "Ireland", "South",
    "Poland", "Scotland", "Thailand", "Ecuador", "Jamaica", "Laos", "Portugal", "Trinadad&Tobago", "Puerto-Rico",
    "Haiti", "El-Salvador", "Honduras", "Vietnam", "Peru", "Nicaragua", "Mexico", "Guatemala", "Columbia",
    "Dominican-Republic", "Holand-Netherlands", "Outlying-US(Guam-USVI-etc)"
]