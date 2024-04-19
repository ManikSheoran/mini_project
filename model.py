import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Enable inline plotting for matplotlib
plt.rcParams['figure.figsize'] = (12, 6)
warnings.simplefilter(action='ignore')

# Load the dataset
df = pd.read_csv('train-data-final.csv')

# Data Preprocessing
df.drop(['Name', 'New_Price', 'Unnamed: 0'], axis=1, inplace=True)

# Dropping rows with empty cells
df.dropna(subset=['Mileage', 'Engine', 'Power', 'Seats'], inplace=True)

# One-hot encoding categorical variables
Location = pd.get_dummies(df[['Location']])
Fuel_t = pd.get_dummies(df[['Fuel_Type']])
Transmission = pd.get_dummies(df[['Transmission']], drop_first=True)

df = pd.concat([df, Location, Fuel_t, Transmission], axis=1)
df.drop(["Location", "Fuel_Type", "Transmission"], axis=1, inplace=True)

# Train Test Split
X = df.loc[:, ['Kilometers_Driven', 'Year', 'Owner_Type', 'Mileage', 'Engine', 'Power',
               'Seats', 'Fuel_Type_CNG', 'Fuel_Type_Diesel', 'Fuel_Type_LPG', 'Fuel_Type_Petrol',
               'Transmission_Manual','Location_Ahmedabad', 'Location_Bangalore',
               'Location_Chennai', 'Location_Coimbatore', 'Location_Delhi',
               'Location_Hyderabad', 'Location_Jaipur', 'Location_Kochi',
               'Location_Kolkata', 'Location_Mumbai', 'Location_Pune']]
y = df.loc[:, ['Price']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Scaling the data for better training
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# CatBoostRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

model = CatBoostRegressor(iterations=6000,
                          learning_rate=0.1,
                          depth=5,
                          loss_function='RMSE',
                          verbose=1000)

model.fit(X_train, y_train, eval_set=(X_test, y_test))
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-Squared value: {r2_test}')

data_frame = pd.DataFrame({'Actual Price': y_test.squeeze(), 'Predicted Price': y_pred.squeeze()})
print(data_frame)

import joblib

# Save the trained model
joblib.dump(model, 'catboost_model.pkl')

# Save the scaler
joblib.dump(sc, 'scaler.pkl')
