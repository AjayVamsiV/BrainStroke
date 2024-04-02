import streamlit as st
import pickle
from encode import encode
import joblib

with open("Nithin/Models/knn_model.pkl", "rb") as f:
  knn_model = pickle.load(f)
with open('Nithin/Models/dtree_model.pkl', 'rb') as f:
  dtree_model = pickle.load(f)
rfmodel = joblib.load('Nithin/Models/ranfor_model.sav')
xgmodel = joblib.load('Nithin/Models/xgboost_model.pkl')
def main():
    st.title("Stroke Prediction")

    # Add dropdowns for each input feature
    gender = st.selectbox('Select Gender', ['Male', 'Female'])
    age = st.number_input('Enter Age', format='%d', value=0)
    hypertension = st.selectbox('Select Hypertension', [0, 1])
    heart_disease = st.selectbox('Select Heart Disease', [1, 0])
    ever_married = st.selectbox('Select Ever Married', ['Yes', 'No'])
    work_type = st.selectbox('Select Work Type', ['Private', 'Self-employed', 'Govt_job', 'children'])
    Residence_type = st.selectbox('Select Residence Type', ['Urban', 'Rural'])
    glucose_level = st.number_input('Enter Glucose Level', format='%f', value=0.0)
    bmi = st.number_input('Enter BMI', format='%f', value=0.0)
    smoking_status = st.selectbox('Select Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])
    # Add a button to trigger the prediction
    if st.button('Predict'):
        preprocessedData = encode(gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, glucose_level, bmi, smoking_status)
        pred1 = dtree_model.predict(preprocessedData)[0]
        pred2 = rfmodel.predict(preprocessedData)[0]
        pred3 = knn_model.predict(preprocessedData)[0]
        pred4 = xgmodel.predict(preprocessedData)[0]
        result = (pred1+pred2+pred3+pred4)/4
        print('output dtree',dtree_model.predict(preprocessedData))
        print('output rf',rfmodel.predict(preprocessedData))
        print('output knn',knn_model.predict(preprocessedData))
        print('output xgboost',xgmodel.predict(preprocessedData))
        if(result<0.5):
           st.success('No Stroke')
        else:
           st.error('Has Stroke')

if __name__ == "__main__":
    main()
