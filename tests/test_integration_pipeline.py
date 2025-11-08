import pytest
import os
from src.mlops.modeling.train import train_model 

@pytest.mark.slow 
def test_full_pipeline_run():
    """
    Prueba de integración (smoke test) para el pipeline completo.
    Verifica que el flujo (carga -> prepro -> entrena -> métricas)
    se ejecuta de principio a fin sin errores.
    """
    
    # ARRANGE
    # Asumimos que los datos de DVC ya están descargados
    data_path = "data/raw/seoul_bike_sharing_modified.csv"
    model_type = "ridge" # Usamos Ridge porque es el más rápido para un test
    
    assert os.path.exists(data_path), f"Datos no encontrados en {data_path}. Ejecuta 'dvc pull' primero."

    # ACT
    # Ejecutamos el pipeline completo
    metrics = None
    try:
        metrics = train_model(
            data_path=data_path,
            target="demanda",
            model_type=model_type,
            log_to_mlflow=False
        )
    except Exception as e:
        pytest.fail(f"El pipeline de integración falló: {e}")

    # ASSERT
    # Verificamos que el pipeline devolvió las métricas
    assert metrics is not None, "La función 'train_model' no devolvió métricas."
    assert "rmse" in metrics, "La métrica 'rmse' no fue calculada."
    assert "r2" in metrics, "La métrica 'r2' no fue calculada."
    assert metrics["rmse"] > 0 # Verifica que es un número válido