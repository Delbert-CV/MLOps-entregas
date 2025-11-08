import pandas as pd
import pytest
from src.mlops.modeling.train import DataCleaner, FeatureEngineering # ¡Importa tus clases!

def test_datacleaner_transform(sample_clean_data):
    """
    Prueba unitaria para la clase DataCleaner.
    Verifica que las columnas se limpien como se espera.
    """
    # ARRANGE
    cleaner = DataCleaner()
    
    # ACT
    # Usamos fit_transform porque es un transformador de Sklearn
    cleaned_df = cleaner.fit_transform(sample_clean_data)
    
    # ASSERT
    assert isinstance(cleaned_df, pd.DataFrame)
   
    assert 'holiday' in cleaned_df.columns 
    assert 'demanda' in cleaned_df.columns 

def test_featureengineering_transform(sample_clean_data):
    """
    Prueba unitaria para la clase FeatureEngineering.
    Verifica que las nuevas features de tiempo se creen.
    """
    # ARRANGE
    feature_eng = FeatureEngineering()
    
    # ACT
    featured_df = feature_eng.fit_transform(sample_clean_data)
    
    # ASSERT
    assert isinstance(featured_df, pd.DataFrame)
    # (Basado en el nombre 'create_time_features')
    assert "month" in featured_df.columns
    assert "day_of_week" in featured_df.columns
    assert "month_sin" in featured_df.columns  
    assert "hour_cos" in featured_df.columns  