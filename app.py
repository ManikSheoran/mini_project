from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('catboost_model.pkl')
scaler = joblib.load('scaler.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    input_data = request.form.to_dict()

    # Prepare the input data as a DataFrame
    input_df = pd.DataFrame({
        'Kilometers_Driven': [float(input_data['Kilometers_Driven'])],
        'Year': [int(input_data['Year'])],
        'Owner_Type': [int(input_data['Owner_Type'])],
        'Mileage': [float(input_data['Mileage'].split()[0])],  # Extract the mileage value
        'Engine': [float(input_data['Engine'].split()[0])],  # Extract the engine value
        'Power': [float(input_data['Power'].split()[0])],  # Extract the power value
        'Seats': [float(input_data['Seats'])],
        'Fuel_Type_CNG': [1 if input_data['Fuel_Type'] == 'CNG' else 0],
        'Fuel_Type_Diesel': [1 if input_data['Fuel_Type'] == 'Diesel' else 0],
        'Fuel_Type_LPG': [1 if input_data['Fuel_Type'] == 'LPG' else 0],
        'Fuel_Type_Petrol': [1 if input_data['Fuel_Type'] == 'Petrol' else 0]
    })

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    return render_template('result.html', prediction=round(prediction[0], 2))


if __name__ == '__main__':
    app.run(debug=True)
