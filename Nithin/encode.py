import pickle
import numpy as np
    
with open('Encoders/standard_scaler.pkl', 'rb') as f:
  scaler = pickle.load(f)

with open('Encoders/label_gender.pkl', 'rb') as file:
    label_gender = pickle.load(file)

with open('Encoders/label_married.pkl', 'rb') as file:
    label_married = pickle.load(file)

with open('Encoders/label_work.pkl', 'rb') as file:
    label_work = pickle.load(file)

with open('Encoders/label_residence.pkl', 'rb') as file:
    label_residence = pickle.load(file)

with open('Encoders/label_smoking.pkl', 'rb') as file:
    label_smoking = pickle.load(file)

print(label_work.transform(['Private']))

def encode(gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, glucose_level, bmi, smoking_status):
    gender = label_gender.transform([gender])[0]
    ever_married = label_married.transform([ever_married])[0]
    work_type = label_work.transform([work_type])[0]
    Residence_type = label_residence.transform([Residence_type])[0]
    smoking_status = label_smoking.transform([smoking_status])[0]
    input = np.array([gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, glucose_level, bmi, smoking_status])
    scaled = scaler.transform(input.reshape(1,10))
    print(scaled.shape,scaled)
    return scaled