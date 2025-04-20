from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import joblib
import os
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from monitoring.logger import setup_logger
from models.evaluation import predict_for_country

# Set up API logger
logger = setup_logger("api_logger", "logs/api.log")

app = FastAPI(title="Business Forecasting API")

# Load the trained model
MODEL_PATH = os.path.join("models", "model_repository", "best_model.pkl")
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None

class PredictionResponse(BaseModel):
    country: str
    predictions: List[Dict[str, Any]]
    model_version: str
    model_type: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Business Forecasting API"}

@app.get("/health")
def health_check():
    """Endpoint to check if the API and model are working properly"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/predict/{country}", response_model=PredictionResponse)
def predict(country: str, horizon: int = Query(12, description="Prediction horizon in months")):
    """
    Get predictions for a specific country
    
    Args:
        country: Country code to get predictions for
        horizon: Number of months to forecast
    """
    logger.info(f"Prediction requested for country: {country}, horizon: {horizon}")
    
    try:
        # Validate country input (convert to uppercase for consistency)
        country = country.upper()
        
        # Get predictions using the model
        predictions, model_info = predict_for_country(country, horizon, model)
        
        # Format response
        response = {
            "country": country,
            "predictions": predictions,
            "model_version": model_info["version"],
            "model_type": model_info["type"]
        }
        
        logger.info(f"Successfully generated predictions for {country}")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating predictions")

@app.get("/predict/", response_model=List[PredictionResponse])
def predict_all(horizon: int = Query(12, description="Prediction horizon in months")):
    """
    Get predictions for all available countries
    
    Args:
        horizon: Number of months to forecast
    """
    logger.info(f"Prediction requested for all countries, horizon: {horizon}")
    
    try:
        # Get list of available countries from dataset
        countries_data = pd.read_csv(os.path.join("data", "processed", "countries.csv"))
        available_countries = countries_data["country_code"].unique().tolist()
        
        all_predictions = []
        for country in available_countries:
            predictions, model_info = predict_for_country(country, horizon, model)
            all_predictions.append({
                "country": country,
                "predictions": predictions,
                "model_version": model_info["version"],
                "model_type": model_info["type"]
            })
        
        logger.info(f"Successfully generated predictions for all {len(available_countries)} countries")
        return all_predictions
        
    except Exception as e:
        logger.error(f"Prediction error for all countries: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating predictions")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
