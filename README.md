# MLOps-entregas
# Seoul Bike Sharing Demand
## Equipo 48

| Nombre | Matrícula |
| ------ | --------- |
| André Zaragoza  | A01797076 |
| Héctor Santillán | A01633395 |
| Pablo de Jesus González | A01321850 |
| Delbert Custodio | A01795613 |
| Abel Diaz | A00566705 |

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

MLOps applied to Seoul Bike sharing demand

## Project Organization

```
├── LICENSE
├── Makefile
├── README.md
├── data
│   ├── external
│   ├── interim
│   ├── processed                               <-- Datasets para modelado
│   │   ├── seoul_bike_sharing_cleaned.csv      <-- No incluido en el repo por razones de seguridad
│   │   ├── seoul_bike_sharing_cleaned.csv.dvc  <-- Incluido en el repo para reproducir experimentos
│   │   ├── seoul_bike_sharing_feature.csv      <-- No incluido en el repo por razones de seguridad
│   │   └── seoul_bike_sharing_feature.csv.dvc  <-- Incluido en el repo para reproducir experimentos
│   └── raw                                     <-- Datasets original y modificado
│       ├── seoul_bike_sharing_modified.csv     <-- No incluido en el repo por razones de seguridad
│       ├── seoul_bike_sharing_modified.csv.dvc <-- Incluido en el repo para reproducir experimentos
│       ├── seoul_bike_sharing_original.csv     <-- No incluido en el repo por razones de seguridad
│       └── seoul_bike_sharing_original.csv.dvc <-- Incluido en el repo para reproducir experimentos
├── docs
├── estructura_proyecto.txt
├── models
│   ├── lightgbm.pkl                            <-- No incluido en el repo por razones de seguridad
│   ├── lightgbm.pkl.dvc                        <-- Incluido en el repo para reproducir experimentos
│   ├── perceptron.pkl                          <-- No incluido en el repo por razones de seguridad
│   ├── perceptron.pkl.dvc                      <-- Incluido en el repo para reproducir experimentos
│   ├── ridge.pkl                               <-- No incluido en el repo por razones de seguridad
│   ├── ridge.pkl.dvc                           <-- Incluido en el repo para reproducir experimentos
│   ├── xgboost.pkl                             <-- No incluido en el repo por razones de seguridad
│   └── xgboost.pkl.dvc                         <-- Incluido en el repo para reproducir experimentos
├── notebooks
│   └── fase1.ipynb                             <-- EDA y Feature engineering inicial
├── pyproject.toml
├── references
├── reports
│   └── figures
├── requirements.txt
└── src
    ├── __init__.py
    ├── main.py
    ├── mlops
    │   ├── __init__.py
    │   ├── dataset.py
    │   ├── features.py
    │   ├── modeling
    │   │   ├── __init__.py
    │   │   └── train.py
    │   └── preprocess.py
    └── mlruns
```

--------
## Pruebas (Testing) - Punto 1

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

## Servicio de API (FastAPI) - Punto 2

Este proyecto expone el modelo entrenado a través de una API RESTful construida con FastAPI, permitiendo que cualquier aplicación externa consuma predicciones.

### Artefacto del Modelo en Producción

El servicio está configurado para cargar el siguiente artefacto, el cual fue entrenado, versionado con DVC y registrado con MLflow:

* **Archivo del Pipeline:** `xgboost.pkl`
* **Ruta de Registro (MLflow):** `models:/xgboost_pipeline_model/Production` (o la ruta de tu mejor modelo)
* **Métrica Clave (RMSE):** 172.65 (del mejor *run* de XGBoost)

### Cómo Probar la API Localmente

1.  Asegúrate de que tu entorno virtual (`venv`) esté activado y que todas las dependencias estén instaladas:
    ```bash
    pip install -r requirements.txt
    ```
2.  Ejecuta el servidor de FastAPI usando `uvicorn`. El comando `--reload` reiniciará el servidor automáticamente si haces cambios en el código.
    ```powershell
    # (Asegúrate de estar en la raíz del proyecto)
    uvicorn app:app --host 127.0.0.1 --port 8000 --reload
    ```
3.  El servidor estará corriendo en `http://127.0.0.1:8000`.

### Documentación del Schema (Swagger UI)

FastAPI genera automáticamente la documentación interactiva de la API (schema de entrada/salida) usando OpenAPI/Swagger.

1.  Con el servidor corriendo, abre tu navegador y ve a:
    **`http://localhost:8000/docs`**

2.  Desde esta página, puedes probar el endpoint `POST /predict` directamente:
    * Haz clic en `POST /predict` para expandirlo.
    * Haz clic en **"Try it out"**.
    * El "Example Value" te mostrará el JSON de entrada requerido.
    * Haz clic en **"Execute"** para enviar la petición y recibir una predicción en vivo.