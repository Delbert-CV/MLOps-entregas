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
    
    '''
    FUNCION DE CARGA DE ARCHIVOS CSV
    '''
    
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
    
    if value_to_clean is None or value_to_clean == '':
        return None
    
    cleaned = str(value_to_clean).strip()
    
    if cleaned.lower() == 'nan':
        return None
    
    return cleaned

def correct_holiday_values( value_to_clean ):
    
    
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
    
    try:
        float(value_to_check)
        return float(value_to_check)
    except:
        return np.nan

def correct_season_values( value_to_clean):
    
    if ( (value_to_clean == 'nan') or (value_to_clean is None ) or (value_to_clean == '')):
        return None
    
    else:
        value_to_clean = str(value_to_clean).strip().lower()
        
        if value_to_clean == 'nan':
            return None
        
        return value_to_clean
    
def correct_hour(x):
    
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
    
    df['date'] = df['date'].apply(correct_date)
    
    return df

def clean_holiday_column(df: pd.DataFrame):
    
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
    
    df['functioning_day_or_not'] = df['functioning_day'].apply(correct_functioning_day_values)
    df['functioning_day'] = df['functioning_day_or_not']
    df.drop(columns=['functioning_day_or_not'], inplace=True)
    
    return df

def clean_seasons(df: pd.DataFrame):
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
    
    df.drop(columns=['mixed_type_col'], inplace= True)
    return df

def clean_weather_features(df: pd.DataFrame):
    
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
    
    df.dropna(subset=['date', 'hour'], inplace=True)
    
    return df

def clean_target(df: pd.DataFrame):
    
    df['demanda'] = df['demanda'].apply(check_if_numeric_value)
    df.dropna(subset=['demanda'], inplace=True)
    
    return df

def clean_hour(df: pd.DataFrame):
    
    df['hour'] = pd.to_numeric(df['hour'], errors='coerce')
    df['hour'] = df['hour'].apply(correct_hour)
    
    return df
    

def dataset_split(df: pd.DataFrame,
                    target: str,
                    test_size: float = 0.2,
                    random_state: int = 42):
    
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