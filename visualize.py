"""
=========================================================
FACIAL EMOTION RECOGNITION — TRAINING VISUALIZATION (L5/L6)
Author: CoreyLeath-code (Corey Leath)
Description:
    - Loads training history (history.json)
    - Plots accuracy & loss curves
    - Computes confusion matrix
    - Saves all graphs into /plots folder
=========================================================
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# =========================================================
# Load training history
# =========================================================
HISTORY_PATH = "history.json"
MODEL_PATH = "emotion_model_final.h5"
CSV_PATH = "fer2013.csv"

if not os.path.exists(HISTORY_PATH):
    raise FileNotFoundError("❌ history.json not found. Train the model first.")

with open(HISTORY_PATH, "r") as f:
    history = json.load(f)

# Create plots directory
os.makedirs("plots", exist_ok=True)

print("📈 Loaded training history.")

# =========================================================
# Plot Accuracy & Loss
# =========================================================
def save_training_plots():
    acc = history["accuracy"]
    val_acc = history["val_accuracy"]
    loss = history["loss"]
    val_loss = history["val_loss"]

    epochs = range(1, len(acc) + 1)

    # Accuracy plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, acc, label="Training Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("plots/accuracy_curve.png")
    plt.close()

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("plots/loss_curve.png")
    plt.close()

    print("📊 Saved accuracy & loss plots in /plots")


# =========================================================
# Load dataset for confusion matrix
# =========================================================
def load_dataset():
    import pandas as pd

    df = pd.read_csv(CSV_PATH)
    pixels = np.array([np.array(p.split(), dtype="float32").reshape(48, 48, 1) for p in df["pixels"]])
    pixels = pixels / 255.0

    labels = df["emotion"].values
    return pixels, labels


# =========================================================
# Confusion Matrix
# =========================================================
def save_confusion_matrix():
    EMOTION_LABELS = [
        "Angry", "Disgust", "Fear",
        "Happy", "Sad", "Surprise", "Neutral"
    ]

    # Load model & data
    model = tf.keras.models.load_model(MODEL_PATH)
    X, y_true = load_dataset()

    # Predict
    preds = model.predict(X)
    y_pred = np.argmax(preds, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=EMOTION_LABELS,
                yticklabels=EMOTION_LABELS)

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("plots/confusion_matrix.png")
    plt.close()

    print("🧩 Saved confusion matrix to /plots")

    # Classification report
    report = classification_report(y_true, y_pred, target_names=EMOTION_LABELS)
    with open("plots/classification_report.txt", "w") as f:
        f.write(report)

    print("📄 Saved classification report to /plots")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    save_training_plots()
    save_confusion_matrix()

    print("\n🎉 Visualization complete! Plots saved in the /plots folder.\n")
