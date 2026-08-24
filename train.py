"""
================================================================================
FACIAL EMOTION RECOGNITION — TRAINING PIPELINE (L6 ENGINEERING STANDARD)
Author: Corey Leath (CoreyLeath-code)

Purpose:
    Train a CNN-based emotion classification model on the FER-2013 dataset.

Architecture Role:
    This script orchestrates:
        - Data loading
        - Preprocessing
        - Train/validation split
        - Model construction
        - Training with callbacks
        - Artifact persistence (model + metrics)

Design Decisions:
    - Uses full in-memory dataset loading (acceptable for FER-2013 size).
    - Sequential CNN chosen for interpretability and reproducibility.
    - Adam optimizer with low LR for stable convergence.
    - EarlyStopping prevents overfitting on small dataset.
    - ModelCheckpoint preserves best validation accuracy weights.

Tradeoffs:
    - In-memory loading increases RAM usage (not scalable to large datasets).
    - Sequential API over Functional API for simplicity.
    - No data augmentation (kept deterministic for baseline reproducibility).

Future Production Improvements:
    - Replace NumPy loading with tf.data pipeline.
    - Add TensorBoard logging.
    - Add mixed precision training.
    - Add data augmentation layer.
================================================================================
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

class Config:
    CSV_PATH = "fer2013.csv"
    RANDOM_SEED = 42
    TEST_SPLIT = 0.1
    NUM_CLASSES = 7
    IMAGE_SIZE = (48, 48, 1)
    EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4


# ==============================================================================
# 2. LOGGING SETUP
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ==============================================================================
# 3. REPRODUCIBILITY
# ==============================================================================

def set_seed(seed: int):
    """
    Ensures deterministic behavior across NumPy and TensorFlow.
    Note: True determinism may require additional GPU-level configuration.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ==============================================================================
# 4. DATA LOADING
# ==============================================================================

def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Loads FER-2013 dataset from CSV.

    Raises:
        FileNotFoundError if dataset is missing.
        ValueError if required columns are absent.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"emotion", "pixels", "Usage"}
    if not required_cols.issubset(df.columns):
        raise ValueError("CSV missing required columns.")

    logger.info("Dataset loaded successfully.")
    return df


# ==============================================================================
# 5. PREPROCESSING
# ==============================================================================

def preprocess_pixels(pixel_string: str) -> np.ndarray:
    """
    Converts FER pixel string into normalized 48x48 grayscale image.

    Design Choice:
        Normalize to [0,1] to stabilize gradient descent.

    Complexity:
        O(n) per sample where n = 48*48.
    """
    pixels = np.array(pixel_string.split(), dtype="float32")
    image = pixels.reshape(Config.IMAGE_SIZE)
    return image / 255.0


def prepare_data(df: pd.DataFrame):
    """
    Converts raw dataframe into train/validation splits.

    Tradeoff:
        Full dataset loaded into memory.
        Acceptable for FER-2013 (~35k images).
    """
    X = np.array(list(map(preprocess_pixels, df["pixels"])))
    y = tf.keras.utils.to_categorical(df["emotion"], Config.NUM_CLASSES)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=Config.TEST_SPLIT,
        random_state=Config.RANDOM_SEED,
        stratify=y
    )

    logger.info(f"Train shape: {X_train.shape}")
    logger.info(f"Validation shape: {X_val.shape}")

    return X_train, X_val, y_train, y_val


# ==============================================================================
# 6. MODEL ARCHITECTURE
# ==============================================================================

def build_model() -> tf.keras.Model:
    """
    Builds CNN architecture.

    Architecture Rationale:
        - Stacked Conv blocks increase representational depth.
        - BatchNorm improves convergence stability.
        - Dropout mitigates overfitting on small dataset.
        - Dense(256) balances expressiveness with parameter count.

    Parameter Count:
        ~3-4M parameters depending on configuration.
    """
    model = models.Sequential([
        layers.Input(shape=Config.IMAGE_SIZE),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(Config.NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ==============================================================================
# 7. TRAINING
# ==============================================================================

def train():
    logger.info("Initializing training pipeline.")
    set_seed(Config.RANDOM_SEED)

    df = load_dataset(Config.CSV_PATH)
    X_train, X_val, y_train, y_val = prepare_data(df)

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "emotion_model_best.h5",
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    logger.info("Starting training.")

    history = model.fit(
        X_train,
        y_train,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    logger.info("Training complete.")

    model.save("emotion_model_final.h5")

    with open("history.json", "w") as f:
        json.dump(history.history, f, indent=4)

    logger.info("Artifacts saved successfully.")


# ==============================================================================
# 8. ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    train()
