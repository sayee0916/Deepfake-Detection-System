🌍 Air Quality Index (AQI) Prediction using Machine Learning
📌 Project Overview

Air pollution is a major environmental and public health concern.
This project aims to predict the Air Quality Index (AQI) using machine learning models based on pollutant concentration levels.

The model analyzes key pollutants such as PM2.5, PM10, NO₂, SO₂, CO, and O₃ to estimate AQI for multiple Indian cities.

The final model is deployed as an interactive web application using Streamlit, allowing users to input pollutant values and obtain AQI predictions in real time.

🎯 Problem Statement

Accurate prediction of AQI helps in:

Environmental monitoring

Public health advisories

Government policy planning

Pollution control measures

The goal of this project is to build a regression-based machine learning model to predict AQI using pollutant features.

📊 Dataset Information

Dataset: India City AQI Dataset

Time Period: 2015 – 2023

Cities: 10 major Indian cities

Total Records: ~32,000+

Features Used
PM2.5
PM10
NO₂
SO₂
CO
O₃
City (encoded)

Target Variable
AQI (Air Quality Index)

⚙️ Data Preprocessing

The following preprocessing steps were performed:

Handling missing values using median imputation

Removing records where AQI was missing

Encoding categorical feature (City)

Time-based train-test split to avoid data leakage

Train Data: 2015 – 2021
Test Data: 2022 – 2023

🔎 Exploratory Data Analysis (EDA)

Key observations:

AQI distribution is right-skewed

PM2.5 shows strongest correlation with AQI

Extreme AQI spikes were observed

Presence of outliers

Relationships between pollutants and AQI are non-linear

🤖 Machine Learning Models Implemented

The following regression models were trained and evaluated:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

XGBoost Regressor

Validation Strategy

TimeSeriesSplit (5 splits)

GridSearchCV for hyperparameter tuning

📈 Model Performance Comparison

| Model             | MAE      | RMSE      | R² Score  |
| ----------------- | -------- | --------- | --------- |
| Linear Regression | 5.07     | 20.24     | 0.689     |
| Decision Tree     | 5.37     | 18.09     | 0.751     |
| Random Forest     | 3.02     | 17.62     | 0.764     |
| XGBoost           | **2.93** | **17.35** | **0.771** |



🏆 Best Model

XGBoost Regressor

Reasons:

Highest R² score

Lowest prediction error

Better generalization on unseen data

📊 Residual Analysis

Residuals mostly centered around 0

Few extreme AQI values increase RMSE

Model performs well for moderate AQI levels

Slight difficulty predicting extreme pollution spikes

📌 Key Insights

PM2.5 is the most influential pollutant

Ensemble models outperform linear models

AQI shows strong non-linear behavior

Time-aware validation improves reliability

Boosting models capture complex pollutant interactions

🌐 Web Application (Deployment)

The trained model is deployed using Streamlit to allow real-time AQI predictions.

Features of Web App

User-friendly interface

Input pollutant values

Select city

Predict AQI instantly

🔗 Live App

https://sayee-aqi-prediction.streamlit.app

🛠️ Technologies Used

Python

Pandas

NumPy

Scikit-learn

XGBoost

Streamlit

Joblib

🚀 Future Improvements

Add weather parameters (temperature, humidity, wind speed)

Incorporate lag features for time-series prediction

Develop real-time AQI forecasting system

Improve predictions for extreme pollution events

👩‍💻 Author  
Sayali Sanjay Chidrawar
