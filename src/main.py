# src/mlops/main.py
import argparse
import logging
from mlops.modeling.train import train_model

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