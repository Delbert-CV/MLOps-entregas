import argparse
import logging
from mlops.modeling.train import train_model

"""
    Ejecuta el pipeline completo de entrenamiento desde la línea de comandos.

    Esta función define y procesa los argumentos necesarios para configurar 
    la ejecución del entrenamiento del modelo de predicción de demanda de bicicletas 
    del proyecto Seoul Bike Sharing. Permite especificar parámetros clave como 
    la ruta del dataset, la variable objetivo, el tipo de modelo y los hiperparámetros 
    de configuración.

    Al ejecutarse como script principal (`__main__`), inicializa el registro 
    de logs, interpreta los argumentos proporcionados por el usuario y llama 
    a la función `train_model()` con los valores correspondientes.

    Parámetros (línea de comandos)
    ------------------------------
    --data_path : str, opcional
        Ruta al archivo CSV del dataset. Por defecto: `"data/raw/seoul_bike_sharing_modified.csv"`.
    --target : str, opcional
        Nombre de la columna objetivo a predecir. Por defecto: `"clean_demanda"`.
    --model_type : str, opcional
        Tipo de modelo a entrenar. Opciones válidas: `"ridge"`, `"xgboost"`, `"lightgbm"`, `"perceptron"`.
        Por defecto: "ridge".
    --alpha : float, opcional
        Parámetro de regularización (solo aplicable para modelos Ridge o Lasso). Por defecto: `1.0`.
    --test_size : float, opcional
        Proporción del conjunto de prueba. Por defecto: `0.3`.
    --random_state : int, opcional
        Semilla para asegurar reproducibilidad. Por defecto: `42`.

    Comportamiento
    --------------
    - Configura el entorno de logging para registrar la ejecución.
    - Analiza los argumentos proporcionados por el usuario.
    - Llama a la función `train_model()` para ejecutar el pipeline de entrenamiento.
    - Registra en el log el inicio y finalización del proceso.

    Uso desde CLI
    -------------
    Ejemplo de ejecución desde la terminal:
        python main.py --model_type lightgbm --test_size 0.25 --alpha 0.8

    Retorna
    -------
    None
        La función no devuelve valores, pero ejecuta y registra el proceso completo de entrenamiento.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ejecuta el pipeline de entrenamiento para el proyecto Seoul Bike Sharing."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw/seoul_bike_sharing_modified.csv",
        help="Ruta al dataset CSV.",
    )

    parser.add_argument(
        "--target",
        type=str,
        default="clean_demanda",
        help="Columna objetivo a predecir.",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="ridge",
        choices=["ridge", "xgboost", "lightgbm", "perceptron"],
        help="Tipo de modelo a entrenar.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Parámetro de regularización (solo Ridge/Lasso).",
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.3,
        help="Proporción del conjunto de test.",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    args = parse_args()

    logging.info(f"Iniciando entrenamiento con modelo: {args.model_type}")

    train_model(
        data_path=args.data_path,
        target=args.target,
        model_type=args.model_type,
        alpha=args.alpha,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    logging.info("Entrenamiento finalizado.")