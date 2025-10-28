import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def clean_date_column(value_to_clean):
    
    if value_to_clean is None or value_to_clean == '':
        return None
    
    cleaned = str(value_to_clean).strip()
    
    if cleaned.lower() == 'nan':
        return None
    
    return cleaned