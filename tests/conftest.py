import pandas as pd
import pytest

@pytest.fixture(scope="module")
def sample_clean_data():
    """
    Crea un DataFrame de prueba idéntico al formato de los
    datos raw
    """
    data = {
        "date": ["01/12/2017", "02/12/2017", "03/12/2017"],
        "demanda": [254.0, 204.0, 173.0],
        "hour": [0.0, 1.0, 2.0],
        "temperature(°c)": [-5.2, -5.5, -6.0],
        "humidity(%)": [37.0, 38.0, 39.0],
        "wind_speed_(m/s)": [2.2, 0.8, 1.0],
        "visibility_(10m)": [2000.0, 2000.0, 2000.0],
        "dew_point_temperature(°c)": [-17.6, -17.6, -17.7],
        "solar_radiation_(mj/m2)": [0.0, 0.0, 0.0],
        "rainfall(mm)": [0.0, 0.0, 0.0],
        "snowfall_(cm)": [0.0, 0.0, 0.0],
        "seasons": ["Winter", "Winter", "Winter"],
        "holiday": ["No Holiday", "No Holiday", "Holiday"],
        "functioning_day": ["Yes", "Yes", "No"],
        "mixed_type_col": ["876", "abc", 123] # datos mezclados
    }
    
    return pd.DataFrame(data)