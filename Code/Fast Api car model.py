with open("app.py", "w") as f:
    f.write("""from fastapi import FastAPI, HTTPException, Form
import joblib
import numpy as np

# Load the trained model and preprocessing artifacts
model = joblib.load("xgboost_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")
scaler = joblib.load("scaler.joblib")

# Initialize FastAPI app
app = FastAPI()

@app.post("/predict")
async def predict_price(
    manufacturer: str = Form(...),
    model_name: str = Form(...),
    prod_year: int = Form(...),
    category: str = Form(...),
    fuel_type: str = Form(...),
    engine_volume: str = Form(...),
    mileage: str = Form(...),
    cylinders: float = Form(...),
    gear_box_type: str = Form(...),
    drive_wheels: str = Form(...),
    doors: str = Form(...)
):
    try:
        # Extract and preprocess categorical data
        manufacturer_encoded = label_encoders['Manufacturer'].transform([manufacturer])[0]
        model_encoded = label_encoders['Model'].transform([model_name])[0]
        category_encoded = label_encoders['Category'].transform([category])[0]
        fuel_type_encoded = label_encoders['Fuel type'].transform([fuel_type])[0]
        gear_box_type_encoded = label_encoders['Gear box type'].transform([gear_box_type])[0]
        drive_wheels_encoded = label_encoders['Drive wheels'].transform([drive_wheels])[0]
        doors_encoded = label_encoders['Doors'].transform([doors])[0]

        # Process engine volume
        engine_volume = float(''.join(filter(str.isdigit, engine_volume)))

        # Process mileage
        mileage = int(mileage.replace(" km", "").replace(",", ""))

        # Feature interaction example
        price_category_interaction = category_encoded * 1000

        # Prepare input data
        processed_data = np.array([[
            manufacturer_encoded,
            model_encoded,
            prod_year,
            category_encoded,
            fuel_type_encoded,
            engine_volume,
            mileage,
            cylinders,
            gear_box_type_encoded,
            drive_wheels_encoded,
            doors_encoded,
            price_category_interaction
        ]])

        # Scale continuous features
        continuous_cols = [5, 6]  # Engine volume, mileage
        processed_data[:, continuous_cols] = scaler.transform(processed_data[:, continuous_cols])

        # Predict the price
        predicted_price = model.predict(processed_data)[0]

        return {"predicted_price": f"{predicted_price:.2f} USD"}

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input value: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")")


""")
    
from pyngrok import ngrok
# Set the authtoken for ngrok
ngrok.set_auth_token("Your Token from ngrok")

# Start ngrok to tunnel the FastAPI server
public_url = ngrok.connect(8000)
print(f"Public URL: {public_url}")

!uvicorn app:app #use colab for this statement
