import mlflow
import logging
import argparse  # <-- AÑADIDO
import numpy as np
import mlflow.sklearn
import lightgbm as lgbm
import pickle
import os  # <-- AÑADIDO
import pandas as pd  
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from mlops.features import create_time_features, encode_categorical_features
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, 
                             mean_absolute_percentage_error, explained_variance_score, 
                             median_absolute_error)
from sklearn.base import BaseEstimator, TransformerMixin
from mlops.preprocess import build_preprocessing_pipeline, delete_outliers
from mlops.dataset import (load_data, dataset_split, clean_holiday_column, 
                           clean_functioning_day, clean_seasons, clean_mixed_type, 
                           clean_weather_features, clean_date_hour, clean_hour, 
                           clean_target, clean_date_column)


# Configuración del Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# --- Clases de Transformación Personalizadas 

class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df = clean_date_column(df)
        df = clean_holiday_column(df)
        df = clean_functioning_day(df)
        df = clean_seasons(df)
        df = clean_mixed_type(df)
        df = clean_weather_features(df)
        df = clean_target(df)
        df = clean_date_hour(df)
        df = delete_outliers(df)
        df = clean_hour(df)
        return df
    
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df = create_time_features(df)
        return df

# --- Función Principal de Entrenamiento (Refactorizada) ---

def train_model(
        data_path: str = "data/raw/seoul_bike_sharing_modified.csv",
        target: str = "demanda", 
        model_type: str = "ridge",
        alpha: float = 1.0,
        test_size: float = 0.3,
        random_state: int = 42,
        log_to_mlflow: bool = True 
) -> dict:

    # Iniciamos el logging
    logging.info("="*80)
    logging.info(f"INICIO DE ENTRENAMIENTO - Modelo: {model_type}")
    logging.info("="*80)
    # ... (Tu logging de parámetros es perfecto) ...
    logging.info(f"Parámetros de configuración:")
    logging.info(f"  - data_path: {data_path}")
    logging.info(f"  - model_type: {model_type}")
    logging.info(f"  - test_size: {test_size}")
    logging.info(f"  - random_state: {random_state}")

    # PASO 2/8: Cargando dataset
    logging.info("-" * 80)
    logging.info("PASO 2/8: Cargando dataset...")
    df = load_data(data_path)
    logging.info(f"Dataset cargado - Shape: {df.shape}")
    
    # Limpiamos el dataset
    logging.info("Aplicando limpieza de datos...")
    cleaner = DataCleaner()
    df_cleaned = cleaner.fit_transform(df)
    
    # PASO 3/8: Aplicando feature engineering
    logging.info("-" * 80)
    logging.info("PASO 3/8: Aplicando feature engineering...")
    FE = FeatureEngineering()
    df_featured = FE.fit_transform(df_cleaned)
    logging.info(f"Features generadas - Shape final: {df_featured.shape}")
    
    # PASO 4/8: Dividiendo dataset en train y test
    logging.info("-" * 80)
    logging.info("PASO 4/8: Dividiendo dataset en train y test...")
    X_train, X_test, y_train, y_test = dataset_split(
        df_featured,
        target=target,         # <-- CORREGIDO: Usa el parámetro 'target'
        test_size=test_size,
        random_state=random_state  # <-- CORREGIDO: Usa 'random_state'
    )
    logging.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # PASO 5/8: Configurando pipeline de preprocesamiento
    # ... (Tu código de features numéricas y categóricas es perfecto) ...
    logging.info("-" * 80)
    logging.info("PASO 5/8: Configurando pipeline de preprocesamiento...")
    numerical_features_to_use = [
        'hour', "temperature(°c)", "humidity(%)", "wind_speed_(m/s)",
        "visibility_(10m)", "dew_point_temperature(°c)", "solar_radiation_(mj/m2)",
        "rainfall(mm)", "snowfall_(cm)", "year", "month", "day_of_week", "day_of_year",
        "month_sin", "month_cos", "day_of_week_sin", "day_of_week_cos", "hour_sin", "hour_cos"
    ]
    categorical_features_to_use = ["seasons", "holiday", "functioning_day"]
    preprocessor = build_preprocessing_pipeline(
        categorical_features=categorical_features_to_use, 
        numerical_features=numerical_features_to_use
    )
    
    # PASO 6/8: Seleccionando y configurando modelo
    # ... (Tu lógica de if/elif para seleccionar el modelo es perfecta) ...
    logging.info("-" * 80)
    logging.info("PASO 6/8: Seleccionando y configurando modelo...")
    model_params = {}
    if model_type.lower() == "ridge":
        model = Ridge(alpha=alpha, random_state=random_state)
        model_params["alpha"] = alpha
    elif model_type.lower() == "xgboost":
        xgb_params = {
            "n_estimators": 2000, "learning_rate": 0.1, "subsample": 0.8,
            "colsample_bytree": 0.8, "random_state": random_state, "tree_method": "hist"
        }
        model = XGBRegressor(**xgb_params)
        model_params.update(xgb_params)
    elif model_type.lower() == "lightgbm":
        lgbm_params = {
            "n_estimators": 800, "learning_rate": 0.05, "num_leaves": 40,
            "max_depth": -1, "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.3, "reg_lambda": 0.5, "min_child_samples": 30,
            "random_state": random_state, "n_jobs": -1, "boosting_type": "gbdt",
        }
        model = lgbm.LGBMRegressor(**lgbm_params)
        model_params.update(lgbm_params)
    elif model_type.lower() == "perceptron":
        mlp_params = {
            "hidden_layer_sizes": (256, 128, 4), "activation": "relu", "solver": "adam",
            "learning_rate": "adaptive", "learning_rate_init": 0.001, "alpha": 0.0005,
            "batch_size": 256, "max_iter": 300, "early_stopping": True,
            "n_iter_no_change": 20, "validation_fraction": 0.15, "shuffle": True,
            "random_state": random_state, "tol": 1e-4, "verbose": False
        }
        model = MLPRegressor(**mlp_params)
        model_params.update(mlp_params)
    else:
        error_msg = f"Modelo no soportado: {model_type}."
        logging.error(error_msg)
        raise ValueError(error_msg)
    logging.info(f"Modelo {model_type} configurado.")

    # PASO 7/8: Construyendo pipeline de Scikit-Learn
    logging.info("-" * 80)
    logging.info("PASO 7/8: Construyendo pipeline de Scikit-Learn...")
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", model),
        ]
    )
    logging.info("Pipeline construido.")

    # PASO 8/8: Entrenando modelo y registrando
    logging.info("-" * 80)
    logging.info("PASO 8/8: Entrenando modelo...")

    # --- INICIO DEL BLOQUE REFACTORIZADO ---
    
    if log_to_mlflow:
        # Si 'log_to_mlflow' es True, inicia sesión y registra todo
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment(f"MLOps-{model_type}-Experiment")
        logging.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        
        with mlflow.start_run(run_name=f"{model_type}_Experiment"):
            logging.info(f"Entrenando {model_type} y registrando en MLflow...")
            model_pipeline.fit(X_train, y_train)
            logging.info("✓ Entrenamiento completado")
            
            y_pred = model_pipeline.predict(X_test)
            
            # Calcular métricas
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            evs = explained_variance_score(y_test, y_pred)
            medae = median_absolute_error(y_test, y_pred)
            mbe = np.mean(y_pred - y_test)

            # Log de Parámetros
            logging.info("Registrando parámetros en MLflow...")
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)
            mlflow.log_params(model_params)

            # Log de Métricas
            logging.info("Registrando métricas en MLflow...")
            metrics_dict = {
                "rmse": rmse, "mae": mae, "mape": mape, "r2": r2,
                "explained_variance": evs, "median_ae": medae, "mean_bias_error": mbe
            }
            mlflow.log_metrics(metrics_dict)

            # Log del Modelo
            artifact_name = f"{model_type}_pipeline_model"
            mlflow.sklearn.log_model(model_pipeline, artifact_path=artifact_name)
            logging.info(f"Modelo guardado en MLflow: {artifact_name}")
            
    else:
        # Si 'log_to_mlflow' es False (para 'pytest'), solo entrena y calcula métricas
        logging.info(f"Entrenando {model_type} (MODO DE PRUEBA - Sin registro en MLflow)...")
        model_pipeline.fit(X_train, y_train)
        logging.info("✓ Entrenamiento completado")
        
        y_pred = model_pipeline.predict(X_test)
        
        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        mbe = np.mean(y_pred - y_test)
        
    # --- FIN DEL BLOQUE REFACTORIZADO ---

    # Imprimir métricas a la consola en cualquier caso
    logging.info("="*80)
    logging.info("MÉTRICAS DE EVALUACIÓN:")
    logging.info(f"  RMSE: {rmse:.4f}")
    logging.info(f"  MAE: {mae:.4f}")
    logging.info(f"  MAPE: {mape:.4f}")
    logging.info(f"  R²: {r2:.4f}")
    # ... (otros prints) ...
    logging.info("="*80)
    
    # Guardamos el modelo con pickle para DVC
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)  # <-- AÑADIDO: Asegura que la carpeta exista
    model_save_path = os.path.join(model_dir, f"{model_type}.pkl")
    
    logging.info(f"Guardando pipeline del modelo en: {model_save_path}")
    with open(model_save_path, 'wb') as file:
        pickle.dump(model_pipeline, file)  # <-- CORREGIDO: Guarda el pipeline

    logging.info(f"Modelo {model_type} guardado en {model_save_path}")
    
    # Devuelve el diccionario de métricas (para 'pytest' y uso futuro)
    return {
        "rmse": rmse,
        "r2": r2,
        "mae": mae,
        "mape": mape,
        "explained_variance": evs,
        "median_ae": medae,
        "mean_bias_error": mbe
    }


# --- Bloque de Ejecución Principal ---
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Script de entrenamiento de modelos para Seoul Bike Sharing.")
    
    parser.add_argument(
        "--model_type",
        type=str,
        default="ridge",
        help="Tipo de modelo a entrenar (ej: ridge, xgboost, lightgbm, perceptron)"
    )
    
    args = parser.parse_args()
    
    logging.info(f"Ejecutando script desde __main__ con modelo: {args.model_type}")
    
    # Llama a la función principal
    # Usará los valores por defecto para todo excepto model_type
    # y log_to_mlflow=True (valor por defecto)
    train_model(model_type=args.model_type)