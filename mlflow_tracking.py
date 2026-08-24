# ============================================================
# MLflow Tracking Utility
# Author: Corey Leath (CoreyLeath-code)
# ============================================================

import mlflow
import mlflow.keras
import os

# Create experiment if not exists
EXPERIMENT_NAME = "Facial_Emotion_Recognition"

def start_mlflow_run(params: dict):
    """
    Initializes MLflow experiment and logs parameters.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)
    run = mlflow.start_run()

    # Log parameters
    for key, value in params.items():
        mlflow.log_param(key, value)

    return run


def log_metrics(metrics_dict: dict, step: int = None):
    """
    Logs metrics during training or evaluation.
    """
    for key, value in metrics_dict.items():
        mlflow.log_metric(key, value, step=step)


def log_model(model, model_name="emotion_model"):
    """
    Logs the trained Keras model to MLflow.
    """
    mlflow.keras.log_model(model, model_name)


def log_artifact(file_path: str):
    """
    Logs files (plots, JSON, etc.)
    """
    if os.path.exists(file_path):
        mlflow.log_artifact(file_path)
