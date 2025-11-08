from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import numpy as np
import logging

'''
Configuración del logging
'''

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

'''
FUNCION DE CARGA DE ARCHIVOS CSV
'''

def load_data(path: str) -> pd.DataFrame:
    
    """
    Carga un archivo CSV y devuelve un DataFrame limpio con nombres de columnas estandarizados.

    Esta función forma parte de la etapa de ingesta de datos del pipeline de Machine Learning.
    Se asegura de que el archivo exista antes de cargarlo, y aplica una limpieza mínima a las
    columnas (espacios, mayúsculas/minúsculas y nombres consistentes).

    Parámetros
    ----------
    path : str
        Ruta completa al archivo CSV que se desea cargar. Puede provenir de la carpeta
        `data/raw/`, `data/interim/` o `data/processed/`.

    Devuelve
    -------
    pd.DataFrame
        DataFrame con los datos cargados y nombres de columnas normalizados.

    Errores
    -------
    FileNotFoundError
        Si la ruta especificada no existe o el archivo no puede ser encontrado.
    """
    
    # Generamos el objeto Path
    path = Path(path)
    
    # Si no existe, devolvemos una exception 
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    # Si existe, generamos y devolvemos el dataframe
    
    df = pd.read_csv(path)
    
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df.rename(columns={'rented_bike_count': 'demanda'}, inplace=True)
    
    logging.info(f"Se cargó el siguiente dataset: {path.name} ✅")
    logging.info(f"Dimensiones del dataset cargado: {df.shape[0]} files y {df.shape[1]} columnas")
    return df


'''
FUNCIONES DE CORRECCION DE VALORES
'''
def clean_date_column(value_to_clean):
    
    """
    Limpia valores de texto correspondientes a fechas eliminando espacios
    innecesarios y valores nulos o no válidos.

    Esta función forma parte de la etapa de limpieza dentro del pipeline de
    preprocesamiento. Su propósito es asegurar que los valores en columnas
    de fecha se mantengan consistentes antes de ser transformados a tipo
    datetime.

    Parámetros
    ----------
    value_to_clean : str o None
        Valor individual (celda) que representa una fecha en formato texto.
        Puede contener espacios, valores vacíos o representaciones no válidas
        como 'nan'.

    Retorna
    -------
    str o None
        Valor de texto limpio y sin espacios si es válido.
        Retorna `None` si el valor es nulo, vacío o contiene 'nan'.
    """
    
    if value_to_clean is None or value_to_clean == '':
        return None
    
    cleaned = str(value_to_clean).strip()
    
    if cleaned.lower() == 'nan':
        return None
    
    return cleaned

def correct_holiday_values( value_to_clean ):
    
    """
    Corrige y estandariza los valores de la columna 'holiday' en el dataset.

    Esta función se encarga de limpiar los valores relacionados con días festivos,
    asegurando consistencia antes de su uso en el pipeline de preprocesamiento.

    Parámetros
    ----------
    value_to_clean : str o None
        Valor individual correspondiente al campo 'holiday'. 
        Puede contener valores como 'holiday', 'no holiday', 'nan' o valores vacíos.

    Retorna
    -------
    str o None
        Retorna 'yes' si el valor corresponde a un día festivo,
        'no' si no lo es, y None si el valor es nulo o inválido.
    """
    
    if ( (value_to_clean == 'nan') or (value_to_clean is None ) or (value_to_clean == '')):
        return None
    else:
        value_to_clean = str(value_to_clean).lower()
        value_to_clean = value_to_clean.strip()
    
        if value_to_clean == 'holiday':
            return 'yes'
        elif value_to_clean == 'no holiday':
            return 'no'
        
def correct_functioning_day_values( value_to_clean):
    
    """
    Corrige y estandariza los valores de la columna 'functioning_day' en el dataset.

    Esta función garantiza la consistencia de los valores que indican si un día fue 
    operativo o no dentro del sistema de préstamo de bicicletas, eliminando valores 
    vacíos o inválido.

    Parámetros
    ----------
    value_to_clean : str o None
        Valor individual correspondiente al campo 'functioning_day'. 
        Puede contener valores como 'yes', 'no', 'nan' o valores vacíos.

    Retorna
    -------
    str o None
        Retorna 'yes' si el valor indica un día operativo,
        'no' si no lo es, y None si el valor es nulo o inválido.
    """

      
    if ( (value_to_clean == 'nan') or (value_to_clean is None ) or (value_to_clean == '')):
        return None
    
    else:
        
        value_to_clean = str(value_to_clean).lower()
        value_to_clean = value_to_clean.strip()
    
        if value_to_clean == 'yes':
            return 'yes'
        else:
            return 'no'



def check_if_numeric_value ( value_to_check):
    
    """
    Verifica si un valor puede ser convertido a numérico y lo transforma a tipo float.

    Esta función se utiliza durante la etapa de limpieza para asegurar que las columnas 
    que deberían contener valores numéricos no incluyan caracteres o valores no válidos. 
    Si la conversión falla, el valor se reemplaza por NaN.

    Parámetros
    ----------
    value_to_check : any
        Valor individual a evaluar. Puede ser numérico o texto.

    Retorna
    -------
    float o numpy.nan
        Retorna el valor convertido a float si es numérico.
        En caso contrario, devuelve np.nan.
    """
    
    try:
        float(value_to_check)
        return float(value_to_check)
    except:
        return np.nan



def correct_season_values( value_to_clean):

    """
    Limpia y estandariza los valores de la columna 'season' en el dataset.

    Esta función elimina espacios innecesarios y convierte el texto a minúsculas 
    para mantener la consistencia de los valores que representan las estaciones del año. 
    También reemplaza valores vacíos o no válidos por None.

    Parámetros
    ----------
    value_to_clean : str o None
        Valor individual correspondiente al campo 'season'. 
        Puede incluir nombres de estaciones, valores vacíos o representaciones de 'nan'.

    Retorna
    -------
    str o None
        Retorna el nombre de la estación en minúsculas si es válido,
        o None si el valor es nulo, vacío o inválido.
    """

    
    if ( (value_to_clean == 'nan') or (value_to_clean is None ) or (value_to_clean == '')):
        return None
    
    else:
        value_to_clean = str(value_to_clean).strip().lower()
        
        if value_to_clean == 'nan':
            return None
        
        return value_to_clean


def correct_hour(x):

    """
    Corrige y normaliza los valores de la columna 'hour' en el dataset.

    Esta función asegura que los valores de hora se mantengan dentro del rango válido (0–23), 
    corrigiendo errores comunes en los datos como valores fuera de rango o mal formateados 
    (por ejemplo, 24 o 100 o 1325). Está diseñada para mantener el formato estandar de horas.

    Parámetros
    ----------
    x : int o float
        Valor numérico que representa una hora. Puede contener errores o formatos incorrectos.

    Retorna
    -------
    int
        Retorna la hora corregida en formato entero entre 0 y 23.
        Si el valor no cumple ninguna condición válida, devuelve 0 por defecto.
    """

    
    if 0 <= x <= 23:
        return int(x)
    
    if x == 24:
        return 0
    
    if x == 100:
        return 10
    
    if x >= 100:
        return int(str(int(x)).zfill(4)[:2])
    
    return 0
    

def correct_date(value_to_clean):
    
    """
    Limpia y estandariza los valores de la columna 'date' en el dataset.

    Esta función elimina espacios en blanco y descarta valores vacíos o inválidos 
    ('nan' o None) para asegurar la consistencia de los datos de fecha antes 
    de su conversión a tipo datetime dentro del pipeline de preprocesamiento.

    Parámetros
    ----------
    value_to_clean : str o None
        Valor individual correspondiente al campo 'date'. 
        Puede contener texto con espacios, valores vacíos o representaciones de 'nan'.

    Retorna
    -------
    str o None
        Retorna la fecha limpia como cadena de texto si es válida,
        o None si el valor es nulo, vacío o inválido.
    """

    
    if value_to_clean is None or value_to_clean == '':
        return None

    cleaned = str(value_to_clean).strip()

    if cleaned.lower() == 'nan':
        return None

    return cleaned


'''
FUNCIONES PARA ELIMINAR NANS
'''

def clean_date_column(df: pd.DataFrame):
    
    """
    Limpia la columna 'date' del DataFrame aplicando la función de corrección correspondiente.

    Esta función estandariza los valores de la columna 'date' eliminando espacios y valores inválidos,
    garantizando consistencia en el formato antes de su conversión a tipo datetime.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene la columna 'date' con posibles valores inconsistentes.

    Retorna
    -------
    pandas.DataFrame
        DataFrame con la columna 'date' corregida y lista para procesamiento posterior.
    """

    
    df['date'] = df['date'].apply(correct_date)
    
    return df

def clean_holiday_column(df: pd.DataFrame):
    
    """
    Limpia y estandariza la columna 'holiday' del DataFrame.

    Esta función aplica la corrección de valores en la columna 'holiday', elimina inconsistencias
    textuales y completa los valores faltantes basándose en una lista predefinida de fechas festivas.
    Garantiza que todos los registros tengan valores válidos ('yes' o 'no').

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene la columna 'holiday' y 'date' para la validación de fechas festivas.

    Retorna
    -------
    pandas.DataFrame
        DataFrame con la columna 'holiday' corregida y estandarizada.
    """

    
    df['Holiday_or_not'] = df['holiday'].apply(correct_holiday_values)
    df['holiday'] = df['Holiday_or_not']
    df.drop(columns=['Holiday_or_not'], inplace=True)
    
    festive_dates = ['01/03/2018', '22/05/2018']

    df['holiday'] = [ 
                    'yes' if  (h is None and d in festive_dates)
                    else 'no' if (h is None and d not in festive_dates)
                    else h for d,h in zip(df['date'], df['holiday'])]
    
    return df

def clean_functioning_day(df: pd.DataFrame):

    """
    Limpia y estandariza la columna 'functioning_day' del DataFrame.

    Esta función normaliza los valores de la columna 'functioning_day' utilizando la función de
    corrección correspondiente, eliminando inconsistencias textuales y valores inválidos.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene la columna 'functioning_day' con posibles valores inconsistentes.

    Retorna
    -------
    pandas.DataFrame
        DataFrame con la columna 'functioning_day' corregida y estandarizada.
    """

    
    df['functioning_day_or_not'] = df['functioning_day'].apply(correct_functioning_day_values)
    df['functioning_day'] = df['functioning_day_or_not']
    df.drop(columns=['functioning_day_or_not'], inplace=True)
    
    return df

def clean_seasons(df: pd.DataFrame):
    
    """
    Limpia y estandariza la columna 'seasons' del DataFrame.

    Esta función corrige los valores inconsistentes de la columna 'seasons', 
    elimina espacios y estandariza el formato textual. Además, completa los 
    valores faltantes asignando una estación basada en listas predefinidas 
    de fechas específicas.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene las columnas 'seasons' y 'date' necesarias 
        para la asignación de estaciones.

    Retorna
    -------
    pandas.DataFrame
        DataFrame con la columna 'seasons' corregida, estandarizada y sin valores nulos.
    """

    df['seasons_corrected'] = df['seasons'].apply(correct_season_values)
    df['seasons'] = df['seasons_corrected']
    df.drop(columns=['seasons_corrected'], inplace=True)
    
    winter_dates = [
    '06/12/2017', '14/12/2017', '16/12/2017', '27/12/2017', '01/01/2018',
    '12/01/2018', '14/01/2018', '17/01/2018', '18/01/2018', '21/01/2018',
    '27/01/2018', '30/01/2018', '05/02/2018', '09/02/2018', '19/02/2018',
    '20/02/2018', '25/02/2018', '26/02/2018', '28/02/2018', '05/02/2018'
    ]

    spring_dates = [
    '02/03/2018', '05/03/2018', '08/03/2018', '22/03/2018', '26/03/2018',
    '27/03/2018', '29/03/2018', '02/04/2018', '04/04/2018', '14/04/2018',
    '16/04/2018', '18/04/2018', '21/04/2018', '22/04/2018', '07/05/2018',
    '11/05/2018', '24/05/2018'
    ]

    summer_dates = [
    '10/06/2018', '18/06/2018', '25/06/2018', '28/06/2018', '29/06/2018',
    '07/07/2018', '08/07/2018', '16/07/2018', '18/07/2018', '19/07/2018',
    '22/07/2018', '25/07/2018', '31/07/2018', '01/08/2018', '02/08/2018',
    '04/08/2018', '07/08/2018', '10/08/2018', '22/08/2018', '29/08/2018'
    ]

    autumn_dates = [
    '02/09/2018', '03/09/2018', '11/09/2018', '13/09/2018', '17/09/2018',
    '18/09/2018', '13/10/2018', '14/10/2018', '17/10/2018', '24/10/2018',
    '04/11/2018', '08/11/2018', '10/11/2018', '13/11/2018', '19/11/2018',
    '20/11/2018', '23/11/2018', '29/11/2018', '30/11/2018'
    ]



    df['seasons'] = [ 
                         'winter' if (s is None and d in winter_dates)
                    else 'spring' if (s is None and d in spring_dates)
                    else 'summer' if (s is None and d in summer_dates)
                    else 'autumn' if (s is None and d in autumn_dates)
                    else s for d,s in zip(df['date'], df['seasons'])]

    return df

def clean_mixed_type(df: pd.DataFrame):
    
    """
    Elimina la columna 'mixed_type_col' del DataFrame.

    Esta función elimina columnas que presentan tipos de datos mixtos o 
    inconsistentes, con el fin de evitar errores durante el preprocesamiento 
    y el modelado.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene la columna 'mixed_type_col' a eliminar.

    Retorna
    -------
    pandas.DataFrame
        DataFrame sin la columna 'mixed_type_col'.
    """

    
    df.drop(columns=['mixed_type_col'], inplace= True)
    return df

def clean_weather_features(df: pd.DataFrame):
    
    """
    Limpia y estandariza las columnas meteorológicas del DataFrame.

    Esta función convierte los valores de las variables climáticas a tipo 
    numérico, reemplaza los valores no válidos por NaN y aplica técnicas de 
    llenado hacia adelante y hacia atrás (forward y backward fill) para 
    garantizar la continuidad de los datos.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene las variables meteorológicas a limpiar.

    Retorna
    -------
    pandas.DataFrame
        DataFrame con las columnas meteorológicas convertidas a formato numérico 
        y sin valores faltantes.
    """

    
    weather_cols = [
    'temperature(°c)', 'humidity(%)', 'wind_speed_(m/s)', 'visibility_(10m)', 'dew_point_temperature(°c)', 'solar_radiation_(mj/m2)', 'rainfall(mm)', 'snowfall_(cm)'
    ]
    
    for col in weather_cols:
        df[ f"{col}" ] = df[ f"{col}"].apply(check_if_numeric_value)
        
    for col in weather_cols:
        df[f"{col}"] = df[f"{col}"].fillna(method= 'ffill')
        df[f"{col}"] = df[f"{col}"].fillna(method= 'bfill')
        
    return df

def clean_date_hour(df: pd.DataFrame):
    
    """
    Elimina las filas con valores nulos en las columnas 'date' u 'hour'.

    Esta función garantiza que el conjunto de datos mantenga únicamente 
    registros con información temporal completa, esencial para el análisis 
    horario de la demanda.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene las columnas 'date' y 'hour'.

    Retorna
    -------
    pandas.DataFrame
        DataFrame sin filas con valores nulos en 'date' u 'hour'.
    """

    
    df.dropna(subset=['date', 'hour'], inplace=True)
    
    return df

def clean_target(df: pd.DataFrame):
    
    """
    Limpia y valida los valores de la columna objetivo 'demanda'.

    Esta función convierte los valores de la demanda a formato numérico y 
    elimina las filas que contengan valores no válidos o nulos, asegurando 
    la integridad de la variable objetivo para el modelado.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene la columna 'demanda'.

    Retorna
    -------
    pandas.DataFrame
        DataFrame con la columna 'demanda' limpia y sin valores nulos.
    """

    
    df['demanda'] = df['demanda'].apply(check_if_numeric_value)
    df.dropna(subset=['demanda'], inplace=True)
    
    return df

def clean_hour(df: pd.DataFrame):
    
    """
    Limpia y corrige los valores de la columna 'hour' en el DataFrame.

    Esta función convierte los valores de la hora a formato numérico, 
    corrige valores fuera de rango mediante la función `correct_hour` y 
    asegura que todos los registros contengan horas válidas entre 0 y 23.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene la columna 'hour' con posibles errores o 
        formatos incorrectos.

    Retorna
    -------
    pandas.DataFrame
        DataFrame con la columna 'hour' corregida y estandarizada.
    """

    
    df['hour'] = pd.to_numeric(df['hour'], errors='coerce')
    df['hour'] = df['hour'].apply(correct_hour)
    
    return df
    

def dataset_split(df: pd.DataFrame,
                    target: str,
                    test_size: float = 0.2,
                    random_state: int = 42):
    
    """
    Divide el dataset en conjuntos de entrenamiento y prueba.

    Esta función separa las variables predictoras (features) y la variable 
    objetivo, generando los subconjuntos de entrenamiento y prueba según 
    la proporción especificada. Además, registra en el log información 
    sobre el tamaño de cada subconjunto.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene las variables independientes y la variable objetivo.
    target : str
        Nombre de la columna objetivo que se desea predecir.
    test_size : float, opcional (por defecto=0.2)
        Proporción del dataset que se destinará al conjunto de prueba.
    random_state : int, opcional (por defecto=42)
        Semilla utilizada para asegurar la reproducibilidad de la división.

    Retorna
    -------
    X_train : pandas.DataFrame
        Subconjunto de entrenamiento con las variables independientes.
    X_test : pandas.DataFrame
        Subconjunto de prueba con las variables independientes.
    y_train : pandas.Series
        Subconjunto de entrenamiento con la variable objetivo.
    y_test : pandas.Series
        Subconjunto de prueba con la variable objetivo.
    """

    
    if target not in df.columns:
        raise ValueError(f"La columna a predecir no se encontro: {target}")
    
    # Features
    X = df.drop(columns=[f"{target}"])
    
    # Objetivo
    y = df[f"{target}"]

    # Generamos el split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    #  Loggeamos el split
    logging.info(
        f"Se llevó a cabo el split: Train={len(X_train)} | Test={len(X_test)} | Target={target} | ✅"
    )
    return X_train, X_test, y_train, y_test