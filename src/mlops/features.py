from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def create_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    
    """
    Genera características temporales a partir de la columna de fecha.

    Esta función descompone la variable de fecha en múltiples componentes 
    temporales como año, mes, día de la semana y día del año. Además, aplica 
    codificación cíclica para capturar la naturaleza periódica de las variables 
    temporales, como mes, día de la semana y hora.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene la columna de fecha.
    date_col : str, opcional (por defecto="date")
        Nombre de la columna que contiene las fechas en formato '%d/%m/%Y'.

    Retorna
    -------
    pandas.DataFrame
        DataFrame con las nuevas columnas temporales y las codificaciones cíclicas añadidas.
    """

    
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    
    # Generamos las columnas temporales
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["day_of_year"] = df[date_col].dt.dayofyear

    # Codificación cíclica
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    logging.info("✅ Se cargaron los features temporales correctamente.")
    return df

def encode_categorical_features(df: pd.DataFrame, categorical_cols=None) -> pd.DataFrame:
    
    """
    Aplica codificación one-hot a las variables categóricas del dataset.

    Esta función transforma las columnas categóricas especificadas en variables 
    binarias mediante la técnica One-Hot Encoding, eliminando la primera categoría 
    para evitar multicolinealidad. Facilita la incorporación de variables categóricas 
    en modelos de Machine Learning.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene las variables categóricas.
    categorical_cols : list, opcional
        Lista de nombres de columnas categóricas a codificar. 
        Por defecto: ["seasons", "holiday", "functioning_day"].

    Retorna
    -------
    pandas.DataFrame
        DataFrame con las columnas categóricas reemplazadas por sus variables codificadas.
    """


    if categorical_cols is None:
        categorical_cols = ["seasons", "holiday", "functioning_day"]

    encoder = OneHotEncoder(drop="first", sparse_output=False)
    encoded = encoder.fit_transform(df[categorical_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index,
    )

    df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

    logging.info(f"Se aplicaron los encoders categoricos a las siguientes columnas: {categorical_cols}")
    return df


def scale_numerical_features(df: pd.DataFrame, numerical_cols=None) -> pd.DataFrame:
    
    """
    Escala las características numéricas utilizando la técnica Min-Max Scaling.

    Esta función normaliza los valores de las columnas numéricas para llevarlos 
    al rango [0, 1], mejorando la estabilidad y el desempeño de los modelos 
    sensibles a la magnitud de las variables.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene las características numéricas a escalar.
    numerical_cols : list, opcional
        Lista de columnas numéricas a escalar. Si no se especifica, 
        se utilizan las variables meteorológicas por defecto.

    Retorna
    -------
    pandas.DataFrame
        DataFrame con las columnas numéricas escaladas en el rango [0, 1].
    """


    if numerical_cols is None:
        numerical_cols = [
            "temperature(°c)",
            "humidity(%)",
            "wind_speed_(m/s)",
            "visibility_(10m)",
            "dew_point_temperature(°c)",
            "solar_radiation_(mj/m2)",
            "rainfall(mm)",
            "snowfall_(cm)",
        ]

    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    logging.info(f"Se aplico scaling a las siguientes columnas: {numerical_cols}")
    return df