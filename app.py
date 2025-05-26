import streamlit as st
import pandas as pd
import joblib

st.title("💓 Prédiction de Maladie Cardiaque")

st.write("Entrez les valeurs pour chaque paramètre médical ci-dessous :")

age = st.slider("Âge", 20, 100, 50)
sex = st.radio("Sexe", [0, 1])
cp = st.selectbox("Type de douleur thoracique (cp)", [0,1,2,3])
trestbps = st.number_input("Pression artérielle au repos (trestbps)", 80, 200, 120)
chol = st.number_input("Cholestérol (chol)", 100, 400, 200)
fbs = st.radio("Glycémie à jeun > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Résultat électrocardiogramme (restecg)", [0, 1, 2])
thalach = st.slider("Fréquence cardiaque maximale (thalach)", 70, 210, 150)
exang = st.radio("Angine induite par l'exercice (exang)", [0, 1])
oldpeak = st.number_input("Dépression ST (oldpeak)", 0.0, 6.0, 1.0)
slope = st.selectbox("Pente ST (slope)", [0,1,2])
ca = st.selectbox("Nombre de vaisseaux principaux (ca)", [0,1,2,3])
thal = st.selectbox("Thalassémie (thal)", [3,6,7])

if st.button("Prédire"):
    data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                        columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'])
    model = joblib.load("model.joblib")
    prediction = model.predict(data)[0]
    st.success(f"🔍 Résultat : {'Présence' if prediction == 1 else 'Absence'} de maladie cardiaque")
