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

