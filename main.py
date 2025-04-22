from flask import Flask, request, render_template, redirect, url_for, session
import numpy as np
import pandas as pd
import pickle
from fuzzywuzzy import process

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Load datasets
sym_des = pd.read_csv("kaggle_dataset/symptoms_df.csv")
precautions = pd.read_csv("kaggle_dataset/precautions_df.csv")
workout = pd.read_csv("kaggle_dataset/workout_df.csv")
description = pd.read_csv("kaggle_dataset/description.csv")
medications = pd.read_csv('kaggle_dataset/medications.csv')
diets = pd.read_csv("kaggle_dataset/diets.csv")
Rf = pickle.load(open('model/RandomForest.pkl', 'rb'))

# Symptom and disease mappings
symptoms_list = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3,
    # [...Add the full symptom list here as you've done before...]
    'yellow_crust_ooze': 131
}
diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis',
    # [...Add full disease mapping here...]
    27: 'Impetigo'
}
symptoms_list_processed = {symptom.replace('_', ' ').lower(): value for symptom, value in symptoms_list.items()}

# Function to correct symptom spelling
def correct_spelling(symptom):
    closest_match, score = process.extractOne(symptom.lower(), symptoms_list_processed.keys())
    if score >= 80:
        return closest_match
    return None

# Function to get additional info
def information(predicted_dis):
    desc = " ".join(description[description['Disease'] == predicted_dis]['Description'].values)
    precs = precautions[precautions['Disease'] == predicted_dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.tolist()
    meds = medications[medications['Disease'] == predicted_dis]['Medication'].values.tolist()
    diet = diets[diets['Disease'] == predicted_dis]['Diet'].values.tolist()
    workout_routine = workout[workout['disease'] == predicted_dis]['workout'].values.tolist()
    return desc, precs, meds, diet, workout_routine

# Prediction function
def predicted_value(patient_symptoms):
    i_vector = np.zeros(len(symptoms_list_processed))
    for s in patient_symptoms:
        i_vector[symptoms_list_processed[s]] = 1
    prediction = Rf.predict([i_vector])[0]
    return diseases_list[prediction]

# Routes

@app.route('/')
def root():
    
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('result'))

@app.route('/result')
def result():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('result.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email == 'admin@example.com' and password == 'password':
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            error = 'Invalid Credentials. Please try again.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    symptoms_input = request.form.get('symptoms')
    if not symptoms_input:
        return render_template('index.html', message="Please enter symptoms.")

    patient_symptoms = [s.strip().lower() for s in symptoms_input.split(',')]
    corrected_symptoms = []
    for symptom in patient_symptoms:
        corrected = correct_spelling(symptom)
        if corrected:
            corrected_symptoms.append(corrected)
        else:
            return render_template('index.html', message=f"Symptom '{symptom}' not found.")

    predicted_disease = predicted_value(corrected_symptoms)
    desc, precs, meds, diet, workout_routine = information(predicted_disease)

    return render_template('result.html', disease=predicted_disease,
                           description=desc,
                           precautions=precs,
                           medications=meds,
                           diet=diet,
                           workout=workout_routine)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
