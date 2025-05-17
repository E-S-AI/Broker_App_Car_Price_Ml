import kagglehub
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Download and read data
path = kagglehub.dataset_download("deepcontractor/car-price-prediction-challenge")
print("Path to dataset files:", path)

df = pd.read_csv(path + "/car_price_prediction.csv")

# Preprocessing
columns_to_drop = ['Wheel', 'Airbags', 'Leather interior', 'Color', 'Levy',"ID"]
df.drop(columns=columns_to_drop, axis=1, inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Encode categorical columns
categorical_cols = ['Manufacturer', 'Model', 'Category', 'Fuel type', 'Drive wheels', 'Gear box type', 'Doors', 'Mileage', 'Engine volume']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoder for later use

# Feature engineering
df['Price_Category_Interaction'] = df['Price'] * df['Category']

# Outlier removal
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Data preparation
X = df.drop('Price', axis=1)
y = df['Price']

scaler = StandardScaler()
continuous_cols = ['Mileage', 'Engine volume']
X[continuous_cols] = scaler.fit_transform(X[continuous_cols])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = XGBRegressor(n_estimators=400, learning_rate=0.1, max_depth=9, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MAE: {mae}, MSE: {mse}, R²: {r2}")

# Save model, scaler, encoders, and feature list
joblib.dump(model, "xgboost_model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(label_encoders, "label_encoders.joblib")
#joblib.dump(X.columns.tolist(), "features.pkl")

print("Model and preprocessing elements saved successfully!")


