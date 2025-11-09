# 1. Usa una imagen base oficial de Python (usa la misma versión de tu venv)
FROM python:3.11-slim

# 2. Establece el directorio de trabajo DENTRO del contenedor
WORKDIR /app

# 3. Añade la variable PYTHONPATH
# Esto le dice a Python que busque módulos en /app (para que 'from src.mlops' funcione)
ENV PYTHONPATH="/app"

# 4. Copia el archivo de requerimientos de inferencia
COPY inference-requirements.txt .

# 5. Instala las dependencias
# --no-cache-dir mantiene la imagen ligera
RUN apt-get update && apt-get install -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r inference-requirements.txt

# 6. Copia tu CÓDIGO FUENTE (el 'src' con tus clases personalizadas)
# Copia el 'src' local al '/app/src' del contenedor
COPY ./src /app/src/

# 7. Copia tu MODELO y tu APP
# (Asegúrate de que 'xgboost.pkl' y 'app.py' estén en tu raíz)
COPY ./xgboost.pkl .
COPY ./app.py .

# 8. Expone el puerto en el que correrá uvicorn DENTRO del contenedor
EXPOSE 8000

# 9. El comando para iniciar la API cuando el contenedor arranque
# Escucha en '0.0.0.0' para aceptar conexiones externas
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]