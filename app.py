import pandas as pd
import pickle
from fastapi import FastAPI
from pydantic import BaseModel, Field
import os
import logging

from src.mlops.modeling.train import DataCleaner, FeatureEngineering
from src.mlops.dataset import (load_data, dataset_split, clean_holiday_column, 
                           clean_functioning_day, clean_seasons, clean_mixed_type, 
                           clean_weather_features, clean_date_hour, clean_hour, 
                           clean_target, clean_date_column)
from src.mlops.preprocess import build_preprocessing_pipeline, delete_outliers
# ----------------------------------------------------

# Configura el logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializa la aplicación FastAPI
app = FastAPI(
    title="API de Prediccion de Renta de Bicicletas",
    description="Servicio MLOps para predecir la demanda de bicicletas en Seul.",
    version="1.0.0"
)

# --- CARGA DEL MODELO ---

MODEL_FILE = "xgboost.pkl" # O el nombre de tu mejor .pkl

try:
    with open(MODEL_FILE, "rb") as f:
        model_pipeline = pickle.load(f)
    logger.info(f"Modelo '{MODEL_FILE}' cargado exitosamente.")
except FileNotFoundError:
    logger.error(f"Error: No se encontró el archivo del modelo '{MODEL_FILE}'.")
    model_pipeline = None

# --- VALIDACIÓN DE ENTRADA (PYDANTIC) ---

class BikeFeatures(BaseModel):
    date: str
    hour: float

    temperature_c: float = Field(..., alias="temperature(°c)")
    humidity_percent: float = Field(..., alias="humidity(%)")
    wind_speed_ms: float = Field(..., alias="wind_speed_(m/s)")
    visibility_10m: float = Field(..., alias="visibility_(10m)")
    dew_point_temp_c: float = Field(..., alias="dew_point_temperature(°c)")
    solar_radiation_mjm2: float = Field(..., alias="solar_radiation_(mj/m2)")
    rainfall_mm: float = Field(..., alias="rainfall(mm)")
    snowfall_cm: float = Field(..., alias="snowfall_(cm)")
    seasons: str
    holiday: str
    functioning_day: str
    
    class Config:
        # Esto genera un ejemplo en la documentación de /docs
        schema_extra = {
            "example": {
                "date": "01/12/2017",
                "hour": 1.0,
                "temperature(°c)": -5.5,
                "humidity(%)": 38.0,
                "wind_speed_(m/s)": 0.8,
                "visibility_(10m)": 2000.0,
                "dew_point_temperature(°c)": -17.6,
                "solar_radiation_(mj/m2)": 0.0,
                "rainfall(mm)": 0.0,
                "snowfall_(cm)": 0.0,
                "seasons": "Winter",
                "holiday": "No Holiday",
                "functioning_day": "Yes"
            }
        }

# --- DEFINICIÓN DEL ENDPOINT ---

@app.get("/")
def read_root():
    """Endpoint para verificar que la API esta viva."""
    return {"status": "OK", "message": "Bienvenido a la API de Prediccioon de Bicicletas"}

@app.post("/predict")
def predict(features: BikeFeatures):
    """
    Endpoint de prediccion. Recibe datos raw, los procesa
    y devuelve la demanda de bicicletas(prediccion).
    """
    if model_pipeline is None:
        return {"error": "Modelo no cargado. Contacte al equipo 48 mlops."}, 500

    try:
        # 1. Convertir la entrada de Pydantic a un DataFrame de Pandas
       
        input_df = pd.DataFrame([features.dict(by_alias=True)])

        
        logger.info("Aplicando DataCleaner...")
        cleaner = DataCleaner()
        df_cleaned = cleaner.fit_transform(input_df) # Usamos fit_transform por si tiene estado
        
        logger.info("Aplicando FeatureEngineering...")
        fe = FeatureEngineering()
        df_featured = fe.fit_transform(df_cleaned)

        # 2. Realizar la Predicción
        
        logger.info("Realizando predicción con el pipeline del modelo...")
        prediction = model_pipeline.predict(df_featured)
        
        # 3. Formatear la Salida
        result = prediction[0]
        logger.info(f"Prediccion generada: {result}")
        
        return {"predicted_rented_bike_count": float(result)}
    
    except Exception as e:
        logger.error(f"Error durante la prediccion: {e}")
        return {"error": str(e)}, 400

# Esta línea permite correr el script directamente con 'python app.py' (para debugear)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)