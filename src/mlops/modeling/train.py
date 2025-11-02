import mlflow
import logging
import argparse
import numpy as np
import mlflow.sklearn
import lightgbm as lgbm
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from mlops.features import create_time_features, encode_categorical_features
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, explained_variance_score, median_absolute_error
from sklearn.base import BaseEstimator, TransformerMixin
from mlops.preprocess import build_preprocessing_pipeline, delete_outliers
from mlops.dataset import load_data, dataset_split,  clean_holiday_column, clean_functioning_day, clean_seasons, clean_mixed_type, clean_weather_features, clean_date_hour
from mlops.dataset import clean_hour, clean_target, clean_date_column
import pickle


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

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


def train_model(
                data_path: str = "data/raw/seoul_bike_sharing_modified.csv",
                target: str = "clean_demanda",
                model_type: str = "ridge",   # Dejamo Ridge como el modelo default
                alpha: float = 1.0,
                test_size: float = 0.3,
                random_state: int = 42) -> dict:

    # Iniciamos el logging
    logging.info("="*80)
    logging.info(f"INICIO DE ENTRENAMIENTO - Modelo: {model_type}")
    logging.info("="*80)
    logging.info(f"Parámetros de configuración:")
    logging.info(f"  - data_path: {data_path}")
    logging.info(f"  - model_type: {model_type}")
    logging.info(f"  - test_size: {test_size}")
    logging.info(f"  - random_state: {random_state}")

    # Modulo dataset.py
    logging.info("-" * 80)
    logging.info("PASO 2/8: Cargando dataset...")
    df = load_data(data_path)
    logging.info(f"Dataset cargado - Shape: {df.shape}")
    
    # Limpiamos el dataset
    logging.info("Aplicando limpieza de datos...")
    cleaner = DataCleaner()
    df_cleaned = cleaner.fit_transform(df)
    
    # Feature Engineer
    logging.info("-" * 80)
    logging.info("PASO 3/8: Aplicando feature engineering...")
    FE = FeatureEngineering()
    df_featured = FE.fit_transform(df_cleaned)
    logging.info(f"Features generadas - Shape final: {df_featured.shape}")
    
    # Creamos el train-test split
    logging.info("-" * 80)
    logging.info("PASO 4/8: Dividiendo dataset en train y test...")
    X_train, X_test, y_train, y_test = dataset_split(
        df_featured, target='demanda', test_size=test_size
    )
    logging.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Features a usar en el entrenamiento y prediccion
    logging.info("-" * 80)
    logging.info("PASO 5/8: Configurando pipeline de preprocesamiento...")
    
    numerical_features_to_use = ['hour', "temperature(°c)", "humidity(%)", "wind_speed_(m/s)",
                                 "visibility_(10m)", "dew_point_temperature(°c)", "solar_radiation_(mj/m2)",
                                 "rainfall(mm)", "snowfall_(cm)", "year", "month", "day_of_week", "day_of_year",
                                 "month_sin", "month_cos", "day_of_week_sin", "day_of_week_cos", "hour_sin", "hour_cos"]
    
    categorical_features_to_use = ["seasons", "holiday","functioning_day"]
    
    logging.info(f"  - Features numéricas: {len(numerical_features_to_use)}")
    logging.info(f"  - Features categóricas: {len(categorical_features_to_use)}")
    
    # Instanciamos nuestra pipeline según el modelo a usar 
    preprocessor = build_preprocessing_pipeline( categorical_features= categorical_features_to_use, 
                                                numerical_features = numerical_features_to_use)
    
    # Usamos una variable para almacenar los hiperparámetros específicos
    logging.info("-" * 80)
    logging.info("PASO 6/8: Seleccionando y configurando modelo...")
    
    model_params = {}

    # Si el parametro de modelo es ridge, usamos Ridge
    if model_type.lower() == "ridge":
        model = Ridge(alpha=alpha, random_state=random_state)
        model_params["alpha"] = alpha # Guardamos el parámetro específico
        
    elif model_type.lower() == "xgboost":
        # Parámetros del XGBoost
        xgb_params = {
            "n_estimators": 2000,
            "learning_rate": 0.1,
            #"max_depth": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
            "tree_method": "hist"
        }
        model = XGBRegressor(**xgb_params)
        model_params.update(xgb_params) # Guardamos todos los parámetros de XGBoost
        
    elif model_type.lower() == "lightgbm":
        lgbm_params = {
            "n_estimators": 500,
            "learning_rate": 0.01,
            "max_depth": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
            "n_jobs": -1,
            "boosting_type": "gbdt",
            "num_leaves": 40,
            "max_depth": -1,
            "learning_rate": 0.05,
            "n_estimators": 800,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.3,
            "reg_lambda": 0.5,
            "min_child_samples": 30,
            "random_state": 42
            #"device_type": "gpu"
        }
        model = lgbm.LGBMRegressor(**lgbm_params)
        model_params.update(lgbm_params)
        
    elif model_type.lower() == "perceptron":
        
        mlp_params = {
        "hidden_layer_sizes": (256, 128, 4),
        "activation": "relu",
        "solver": "adam",
        "learning_rate": "adaptive",
        "learning_rate_init": 0.001,
        "alpha": 0.0005,  # regularización L2
        "batch_size": 256,
        "max_iter": 300,
        "early_stopping": True,
        "n_iter_no_change": 20,
        "validation_fraction": 0.15,
        "shuffle": True,
        "random_state": 42,
        "tol": 1e-4,
        "verbose": False
        }
        
        model = MLPRegressor(**mlp_params)
        model_params.update(mlp_params)
        
    else:
        error_msg = f"Modelo no soportado: {model_type}. Opciones válidas: ridge, xgboost, lightgbm, perceptron"
        logging.error(error_msg)
        raise ValueError(error_msg)

    logging.info(f"Modelo {model_type} configurado con {len(model_params)} hiperparámetros")
    
    # Seteamos nuestro pipeline
    
    logging.info("-" * 80)
    logging.info("PASO 7/8: Construyendo pipeline de Scikit-Learn...")
    
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", model),
        ]
    )
    
    logging.info("Pipeline construido:")
    logging.info(f"  Paso 1: preprocessor ({type(preprocessor).__name__})")
    logging.info(f"  Paso 2: regressor ({type(model).__name__})")
    
    logging.info("-" * 80)
    logging.info("PASO 8/8: Entrenando modelo y registrando en MLflow...")

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(f"MLOps-{model_type}-Experiment")
    logging.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    logging.info(f"MLflow Experiment: MLOps-{model_type}-Experiment")
    
    with mlflow.start_run(run_name=f"{model_type}_Experiment"):

        # Hacemos el log del nombre del modelo y los hiperparámetros de manera dinámica
        logging.info(f"Entrenando modelo {model_type}:")
        model_pipeline.fit(X_train, y_train)
        logging.info("✓ Entrenamiento completado")

        logging.info("Realizando predicciones en test set...")
        y_pred = model_pipeline.predict(X_test)
        
        logging.info("Calculando métricas de evaluación...")
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        mbe = np.mean(y_pred - y_test)

        # Parámetros generales
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_numerical_features", len(numerical_features_to_use))
        mlflow.log_param("n_categorical_features", len(categorical_features_to_use))

        # Parámetros específicos del modelo (Ridge: alpha, XGBoost: n_estimators, etc.)
        mlflow.log_params(model_params)

        # Métricas
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("explained_variance", evs)
        mlflow.log_metric("median_ae", medae)
        mlflow.log_metric("mean_bias_error", mbe)

        # Guardamos el modelo con nombre dinámico
        artifact_name = f"{model_type}_pipeline_model"
        mlflow.sklearn.log_model(model_pipeline, artifact_path=artifact_name)
        logging.info(f"Modelo guardado en MLflow: {artifact_name}")

        # Resultados
        logging.info("="*80)
        logging.info("MÉTRICAS DE EVALUACIÓN:")
        logging.info(f"  RMSE: {rmse:.4f}")
        logging.info(f"  MAE: {mae:.4f}")
        logging.info(f"  MAPE: {mape:.4f}")
        logging.info(f"  R²: {r2:.4f}")
        logging.info(f"  Explained Variance: {evs:.4f}")
        logging.info(f"  Median AE: {medae:.4f}")
        logging.info(f"  Mean Bias Error: {mbe:.4f}")
        logging.info("="*80)
        logging.info("✓ Entrenamiento completado exitosamente")
        logging.info("✓ Resultados registrados en MLflow")
        logging.info("="*80)
        
        # Guardamos el modelo usando pickle para versionarlo con dvc
        with open(f'models/{model_type}.pkl', 'wb') as file:
            pickle.dump(model_type, file)

        print(f"Modelo {model_type} guardado en models/model_type.pkl")
    
    return {
        "rmse": rmse,
        "r2": r2,
        "mae": mae,
        "mape": mape,
        "explained_variance": evs,
        "median_ae": medae,
        "mean_bias_error": mbe
    }