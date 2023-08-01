#Import Python Libraries

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


data = pd.read_csv('Crop and fertilizer dataset.csv')


def perform_encoding(df, columns, encoder=None):
    if encoder is None:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoder.fit(df[columns])
    encoded_data = pd.DataFrame(encoder.transform(df[columns]))
    feature_names = encoder.get_feature_names_out(columns)
    encoded_data.columns = feature_names
    return encoded_data, encoder


X_categorical = data[['District_Name', 'Soil_color']]
X_numerical = data[['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature']]
y = data[['Crop', 'Fertilizer']]

categorical_columns = ['District_Name', 'Soil_color']
X_categorical_encoded, encoder = perform_encoding(X_categorical, categorical_columns)

X_encoded = pd.concat([X_categorical_encoded, X_numerical], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
# prediction = model.predict()

import pickle
with open('model.pkl','wb') as f:
    pickle.dump(model,f)
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
