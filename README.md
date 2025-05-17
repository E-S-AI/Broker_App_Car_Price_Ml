# Car Price Prediction API

This project trains an XGBoost model to predict car prices based on various features such as manufacturer, model, fuel type, mileage, etc. It exposes a FastAPI endpoint to make predictions based on user input.

## 🧠 Model Training

The dataset is from [Kaggle: Car Price Prediction Challenge](https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge).

### Steps
- Data cleaning and preprocessing
- Label encoding for categorical features
- Feature engineering and outlier removal
- Scaling numerical features
- Model training using `XGBRegressor`
- Saving model and preprocessing objects with `joblib`

## 🚀 API (FastAPI)

### Endpoint

`POST /predict`

### Form Parameters

- `manufacturer`: str
- `model_name`: str
- `prod_year`: int
- `category`: str
- `fuel_type`: str
- `engine_volume`: str (e.g., "2.0")
- `mileage`: str (e.g., "100,000 km")
- `cylinders`: float
- `gear_box_type`: str
- `drive_wheels`: str
- `doors`: str

### Response

```json
{
  "predicted_price": "8200.75 USD"
}

🧪 Run Locally
uvicorn app:app --reload

To expose the app via a public URL using Ngrok (for Colab):
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_TOKEN")
print(ngrok.connect(8000))

🗃 Files
xgboost_model.joblib — Trained model

scaler.joblib — Scaler for numeric features

label_encoders.joblib — Dictionary of label encoders

app.py — FastAPI app

requirements.txt — Required dependencies

📦 Requirements
pip install -r requirements.txt

