import mlflow
import mlflow.sklearn
import logging
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from mlops.features import create_time_features
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from mlops.preprocess import build_preprocessing_pipeline, delete_outliers
from mlops.dataset import load_data, clean_categorical_colums, clean_numerical_columns, clean_temporal_columns, dataset_split


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
        df = clean_categorical_colums(df)
        df = clean_numerical_columns(df)
        df = clean_temporal_columns(df)
        df = create_time_features(df)
        df = delete_outliers(df)
        return df


def train_model(
    data_path: str = "data/raw/seoul_bike_sharing_modified.csv",
    target: str = "clean_demanda",
    alpha: float = 1.0,
    test_size: float = 0.3,
    random_state: int = 42,
    ):

    # Iniciamos el logging
    logging.info("Iniciando entrenamiento")

    # Modulo dataset.py
    df = load_data(data_path)

    # Limpiamos el dataset
    cleaner = DataCleaner()
    df_cleaned = cleaner.fit_transform(df)
    
    # Creamos el train-test split
    X_train, X_test, y_train, y_test = dataset_split(
        df_cleaned, target='clean_demanda', test_size=test_size, random_state=random_state
    )

    # Instanciamos nuestra pipeline para 
    preprocessor = build_preprocessing_pipeline()

    ridge_model = Ridge(alpha=alpha, random_state=random_state)

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", ridge_model),
        ]
    )

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("MLOps-RidgeModel")
    print("Current tracking URI:", mlflow.get_tracking_uri())

    with mlflow.start_run(run_name="Ridge_Experiment_Fase2"):

        logging.info("Entrenando modelo Ridge...")
        model_pipeline.fit(X_train, y_train)

        y_pred = model_pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mlflow.log_param("model_type", "Ridge")
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(model_pipeline, artifact_path="ridge_pipeline_model")

        logging.info(f"Entrenamiento completado | RMSE={rmse:.3f} | R²={r2:.3f}")
        logging.info("Resultados registrados en MLflow correctamente.")

    return {"rmse": rmse, "r2": r2}


if __name__ == "__main__":
    results = train_model()
    print(f"\n\n\nMétricas finales del modelo: {results}")