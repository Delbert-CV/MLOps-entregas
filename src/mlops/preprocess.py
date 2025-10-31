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
    "snowfall_(cm)",
]

CATEGORICAL_FEATURES = [
    "seasons",
    "holiday",
    "functioning_day",
]


def IQR( df, col ):
    
    q1 = df[f"{col}"].quantile(0.25)
    q3 = df[f"{col}"].quantile(0.75)
    IQR = q3 - q1
    
    lower_limit = q1 - 1.5 * IQR
    upper_limit = q3 + 1.5 * IQR
    
    return [ lower_limit, upper_limit]

def delete_outliers(df: pd.DataFrame) -> pd.DataFrame:
    
    mask_temp   = (df['temperature(°c)'] >= -25.3) & (df['temperature(°c)'] <= 51.5)
    mask_hum    = (df['humidity(%)'] >= 0.0) & (df['humidity(%)'] <= 100.0)
    mask_wind   = (df['wind_speed_(m/s)'] >= 0.00) & (df['wind_speed_(m/s)'] <= 4.5)
    mask_vis    = (df['visibility_(10m)'] >= 0.0) & (df['visibility_(10m)'] <= 3581.0)
    mask_dew    = (df['dew_point_temperature(°c)'] >= -34.0) & (df['dew_point_temperature(°c)'] <= 44.4)
    mask_solar  = (df['solar_radiation_(mj/m2)'] >= 0.00) & (df['solar_radiation_(mj/m2)'] <= 2.4)
    mask_demand = (df['clean_demanda'] >= 0.00) & (df['clean_demanda'] <= 2392.5)
    mask_snow   = (df['snowfall_(cm)'] >= 0.00) & (df['snowfall_(cm)'] <= 40.00)
    mask_rain   = (df['rainfall(mm)'] >= 0.00) & (df['rainfall(mm)'] <= 2015)

    combined_mask = (
        mask_temp &
        mask_hum &
        mask_wind &
        mask_vis &
        mask_dew &
        mask_solar &
        mask_demand &
        mask_snow &
        mask_rain
    )

    df = df[combined_mask]
    
    return df


def build_preprocessing_pipeline(
    numerical_features: list[str] = None,
    categorical_features: list[str] = None,
) -> ColumnTransformer:

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

    logging.info("Configuracion del pipeline:")
    for name, transformer, cols in preprocessor.transformers:
        logging.info(f"  • {name.upper()} → {len(cols)} columns → {cols}")