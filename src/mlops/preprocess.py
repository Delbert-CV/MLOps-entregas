import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

NUMERICAL_FEATURES = [
    "temperature(°c)",
    "humidity(%)",
    "wind_speed_(m/s)",
    "visibility_(10m)",
    "dew_point_temperature(°c)",
    "solar_radiation_(mj/m2)",
    "rainfall(mm)",
    "snowfall_(cm)"
]

CATEGORICAL_FEATURES = [
    "seasons",
    "holiday",
    "functioning_day"
]


def IQR( df, col ):
    
    """
    Calcula los límites inferior y superior para detección de valores atípicos mediante el método del IQR.

    Esta función utiliza el rango intercuartílico (IQR) para identificar posibles 
    valores atípicos en una columna numérica, devolviendo los límites usados 
    para filtrar los datos fuera del rango esperado.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene la columna a evaluar.
    col : str
        Nombre de la columna sobre la cual se calculará el IQR.

    Retorna
    -------
    list[float]
        Lista con dos elementos: [límite_inferior, límite_superior].
    """

    
    q1 = df[f"{col}"].quantile(0.25)
    q3 = df[f"{col}"].quantile(0.75)
    IQR = q3 - q1
    
    lower_limit = q1 - 1.5 * IQR
    upper_limit = q3 + 1.5 * IQR
    
    return [ lower_limit, upper_limit]

def delete_outliers(df: pd.DataFrame) -> pd.DataFrame:
    
    """
    Elimina valores atípicos de las principales variables del dataset.

    Esta función aplica condiciones de rango específicas para cada variable 
    meteorológica y la variable objetivo 'demanda', eliminando registros 
    que contengan valores fuera de los límites definidos.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene las variables meteorológicas y de demanda.

    Retorna
    -------
    pandas.DataFrame
        DataFrame filtrado sin valores atípicos según los rangos establecidos.
    """

    
    mask_temp   = (df['temperature(°c)'] >= -25.3) & (df['temperature(°c)'] <= 51.5)
    mask_hum    = (df['humidity(%)'] >= 0.0) & (df['humidity(%)'] <= 100.0)
    mask_wind   = (df['wind_speed_(m/s)'] >= 0.00) & (df['wind_speed_(m/s)'] <= 4.5)
    mask_vis    = (df['visibility_(10m)'] >= 0.0) & (df['visibility_(10m)'] <= 3581.0)
    mask_dew    = (df['dew_point_temperature(°c)'] >= -34.0) & (df['dew_point_temperature(°c)'] <= 44.4)
    mask_solar  = (df['solar_radiation_(mj/m2)'] >= 0.00) & (df['solar_radiation_(mj/m2)'] <= 2.4)
    
    mask_snow   = (df['snowfall_(cm)'] >= 0.00) & (df['snowfall_(cm)'] <= 40.00)
    mask_rain   = (df['rainfall(mm)'] >= 0.00) & (df['rainfall(mm)'] <= 2015)

    combined_mask = (
        mask_temp &
        mask_hum &
        mask_wind &
        mask_vis &
        mask_dew &
        mask_solar &
       
        mask_snow &
        mask_rain
    )
    
    if 'demanda' in df.columns:
        mask_demand = (df['demanda'] >= 0.00) & (df['demanda'] <= 2392.5)
        combined_mask = combined_mask & mask_demand
    else:
        logging.info('no se detecto columna demanda')

    df = df[combined_mask]
    
    return df


def build_preprocessing_pipeline(
                                    numerical_features: list[str] = None,
                                    categorical_features: list[str] = None) -> ColumnTransformer:
    
    """
    Construye un pipeline de preprocesamiento para datos categóricos y numéricos.

    Esta función crea un objeto `ColumnTransformer` que combina transformaciones 
    personalizadas para columnas categóricas y numéricas, incluyendo codificación 
    One-Hot y escalado estándar. Permite automatizar el preprocesamiento de datos 
    antes del entrenamiento del modelo.

    Parámetros
    ----------
    numerical_features : list[str], opcional
        Lista de nombres de columnas numéricas a escalar. Si no se especifica, 
        se usan las variables definidas en NUMERICAL_FEATURES.
    categorical_features : list[str], opcional
        Lista de nombres de columnas categóricas a codificar. Si no se especifica, 
        se usan las variables definidas en CATEGORICAL_FEATURES.

    Retorna
    -------
    sklearn.compose.ColumnTransformer
        Objeto configurado con las transformaciones para columnas numéricas y categóricas.
    """


    if numerical_features is None:
        numerical_features = NUMERICAL_FEATURES

    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    logging.info("Creando pipeline de pre-procesamiento")

    # Definimos los transformadores por cada tipo de columna
    categorical_transformer = OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    )
    numerical_transformer = StandardScaler()

    # Usamos un columnTransformer
    preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_transformer, categorical_features),
        ("numerical", numerical_transformer, numerical_features),
        ]
        )

    logging.info(
        f"Se creo una pipeline de pre-procesamiento con {len(numerical_features)} features numericos."
        f"y {len(categorical_features)} features categoricos."
    )

    return preprocessor

def summarize_pipeline(preprocessor: ColumnTransformer):
    
    """
    Muestra un resumen de la configuración del pipeline de preprocesamiento.

    Esta función registra en el log la estructura del `ColumnTransformer`, 
    indicando qué transformadores se aplican, cuántas columnas afectan 
    y cuáles son sus nombres, facilitando la trazabilidad del pipeline.

    Parámetros
    ----------
    preprocessor : sklearn.compose.ColumnTransformer
        Objeto del pipeline de preprocesamiento previamente creado.

    Retorna
    -------
    None
        Solo registra la información del pipeline en el log.
    """


    logging.info("Configuracion del pipeline:")
    for name, transformer, cols in preprocessor.transformers:
        logging.info(f"  • {name.upper()} → {len(cols)} columns → {cols}")