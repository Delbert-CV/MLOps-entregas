# 1. Usa una imagen base oficial de Python (usa la misma versión de tu venv)
FROM python:3.11-slim

# 2. Establece el directorio de trabajo DENTRO del contenedor
WORKDIR /app

# 3. Añade la variable PYTHONPATH
# Esto le dice a Python que busque módulos en /app (para que 'from src.mlops' funcione)
ENV PYTHONPATH="/app"

# 4. Copia el archivo de requerimientos de inferencia
COPY inference-requirements.txt .

# 5. Instala las dependencias necesarias
RUN apt-get update && apt-get install -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r inference-requirements.txt

# 6. Copia el código fuente
COPY ./src /app/src/

# 7. Copia el modelo y la aplicación principal
# ⬇️ Aquí estaba el error: debe ser ./models, no .models
COPY ./models/xgboost.pkl /app/xgboost.pkl
COPY ./app.py /app/app.py

# 8. Expone el puerto en el que correrá uvicorn dentro del contenedor
EXPOSE 8000

# 9. Comando para iniciar la API al arrancar el contenedor
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]