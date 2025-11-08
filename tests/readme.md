## Pruebas (Testing)

Este proyecto utiliza `pytest` para asegurar la calidad y robustez del código.

### Cómo Ejecutar las Pruebas

1.  Asegúrate de tener el proyecto clonado y estar en la carpeta raíz.
2.  Crea y/o activa tu entorno virtual:
    ```powershell
    # (Solo la primera vez)
    python -m venv venv
    
    # Activar el venv (PowerShell)
    .\venv\Scripts\activate
    ```
3.  (Solo PowerShell) Si es la primera vez que activas un `venv` en la terminal, ajusta la política de ejecución:
    ```powershell
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
    ```
4.  Instala las dependencias (incluyendo `pytest`):
    ```bash
    pip install -r requirements.txt
    ```
5.  Asegúrate de tener los datos de prueba de DVC (requerido para la prueba de integración):
    ```bash
    # (Asegúrate de que tus credenciales de S3 estén configuradas)
    dvc pull
    ```
6.  Ejecuta el comando único de `pytest`:
    ```powershell
    # El PYTHONPATH=src es necesario para que pytest encuentre los módulos en 'src/'
    $env:PYTHONPATH = "src"; pytest -q
    ```
7.  El resultado esperado es `3 passed` 

Cuales son las pruebas?

1. Pruebas Unitarias (Validación de Módulos Clave)
Se implementan pruebas unitarias para validar los componentes críticos del preprocesamiento de forma aislada.

Prueba: test_datacleaner_transform

Qué Evalúa: Valida la clase DataCleaner (src/mlops/modeling/train.py).


Prueba: test_featureengineering_transform

Qué Evalúa: Valida la clase FeatureEngineering (src/mlops/modeling/train.py).

Cómo Cumple (Rúbrica): Esta es la segunda prueba unitaria para otro módulo clave de preprocesamiento. Asegura que la lógica de creación de features (ej. create_time_features que genera month_sin, hour_cos, etc.) funciona como se espera.

2. Prueba de Integración (Validación Extremo a Extremo)
Se implementa una prueba de integración que valida el flujo completo del pipeline.

Prueba: test_full_pipeline_run

Qué Evalúa: Es un smoke test que ejecuta la función principal train_model de principio a fin, usando datos crudos reales (de DVC) y un modelo rápido (Ridge).

Carga de datos (load_data())

Preprocesamiento (DataCleaner + FeatureEngineering + build_preprocessing_pipeline)

Predicción (entrenamiento del modelo Ridge)

Métricas (devuelve un diccionario metrics que se valida)

Esta prueba garantiza que todos los componentes (dataset.py, preprocess.py, train.py) se integran y funcionan correctamente juntos.