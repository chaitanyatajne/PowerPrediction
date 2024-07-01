import pandas as pd
import streamlit as st
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import StandardScaler
import datetime

import streamlit as st

# Define the background image URL (replace with your image URL)
background_image_url = "https://unsplash.com/photos/photo-of-outer-space-Q1p7bh3SHj8"

# Set the background image using CSS
st.markdown(f"""
    <style>
        .reportview-container {{
            background: url("{background_image_url}");
        }}
    </style>
""", unsafe_allow_html=True)


df=pd.read_csv("powerconsumption.csv")
df=df.drop(['PowerConsumption_Zone1'],axis=1)
df=df.drop(['PowerConsumption_Zone2'],axis=1)

df['Datetime']=pd.to_datetime(df.Datetime)
df.sort_values(by='Datetime', ascending=True, inplace=True)

chronological_order = df['Datetime'].is_monotonic_increasing

time_diffs = df['Datetime'].diff()
equidistant_timestamps = time_diffs.nunique() == 1

def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['day'] = df.index.month
    df['year'] = df.index.year
    df['season'] = df['month'] % 12 // 3 + 1
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week

    # Additional features
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_month_start'] = (df['dayofmonth'] == 1).astype(int)
    df['is_month_end'] = (df['dayofmonth'] == df.index.days_in_month).astype(int)
    df['is_quarter_start'] = (df['dayofmonth'] == 1) & (df['month'] % 3 == 1).astype(int)
    df['is_quarter_end'] = (df['dayofmonth'] == df.groupby(['year', 'quarter'])['dayofmonth'].transform('max'))

    # Additional features
    df['is_working_day'] = df['dayofweek'].isin([0, 1, 2, 3, 4]).astype(int)
    df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
    df['is_peak_hour'] = df['hour'].isin([8, 12, 18]).astype(int)

    # Minute-level features
    df['minute_of_day'] = df['hour'] * 60 + df['minute']
    df['minute_of_week'] = (df['dayofweek'] * 24 * 60) + df['minute_of_day']

    return df.astype(float)

df = df.set_index('Datetime')
df = create_features(df)

from sklearn.preprocessing import StandardScaler

# Separate the input features (X) and target variables (y)
X = df.drop(['PowerConsumption_Zone3'], axis=1)
y=df['PowerConsumption_Zone3']


# Initialize StandardScaler for y
scaler_x = StandardScaler()

# Fit and transform  y
X = scaler_x.fit_transform(X)
adam = optimizers.Adam(0.01)
model_mlp =Sequential()
model_mlp.add(Dense(units=64, activation='relu', input_dim=X.shape[1],kernel_regularizer=regularizers.l2(0.2)))
model_mlp.add(Dense(units=32,activation='relu'))
model_mlp.add(Dropout(0.2))
model_mlp.add(Dense(1,activation='linear'))
model_mlp.compile(loss='mse', optimizer=adam,metrics=['mae'])
model_mlp.summary()
model_mlp.fit(X, y,epochs=50, verbose=2)


    
st.title("Power Consumption Prediction App")
st.sidebar.header("Input Parameters")

import streamlit as st
import datetime

# Date input widget
min_date = datetime.date(2017, 1, 1)
selected_date = st.date_input("Select a date", value=min_date,format="DD/MM/YYYY",min_value=min_date)

# Time input widget
selected_time = st.time_input("Select a time")

# Combine date and time into a single variable
combined_datetime = datetime.datetime.combine(selected_date, selected_time)

# Display the combined datetime
st.write(f"Selected datetime: {combined_datetime}")


# Input widgets for other parameters
temperature = st.number_input("Temperature (°C)", min_value=3.247, max_value=40.0, value=3.247, step=0.0001, format="%.4f")
humidity = st.number_input("Humidity (%)", min_value=3.247, max_value=94.80, value=3.247, step=0.0001, format="%.4f")
windspeed = st.number_input("Windspeed (km/h)", min_value=0.05, max_value=6.483, value=0.05, step=0.0001, format="%.4f")
generaldiffuseflows = st.number_input("General Diffuse Flows", min_value=0.004, max_value=1122.0, value=0.004, step=0.001, format="%.3f")
diffuseflows = st.number_input("Diffuse Flows", min_value=0.019, max_value=933.0, value=0.019, step=0.001, format="%.3f")
# Create a dictionary with user inputs
input_dict = {
    "combined_datetime": [combined_datetime],
    "Temperature": [temperature],
    "Humidity": [humidity],
    "Windspeed": [windspeed],
    "General Diffuse Flows": [generaldiffuseflows],
    "Diffuse Flows": [diffuseflows]
}

# Create a DataFrame from the dictionary
input_data = pd.DataFrame(input_dict)

input_data['combined_datetime']=pd.to_datetime(input_data.combined_datetime)
input_data.sort_values(by='combined_datetime', ascending=True, inplace=True)

chronological_order = input_data['combined_datetime'].is_monotonic_increasing

time_diffs = input_data['combined_datetime'].diff()
equidistant_timestamps = time_diffs.nunique() == 1
input_data = input_data.set_index('combined_datetime')

input_data=create_features(input_data)
input_data=scaler_x.fit_transform(input_data)

# Now use the input_data for prediction
predicted_power = model_mlp.predict(input_data)
predicted_value = predicted_power[0]

# Display predictions
st.write(f"Predicted power consumption: {predicted_value:} units")
