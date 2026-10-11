import csv
from pathlib import Path
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "outputs" /"results"

history_path = RESULTS_DIR / "E01_history.csv"

epochs = []
train_loss = []
val_loss = []
train_acc = []
val_acc = []

with open(history_path, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        epochs.append(int(row["epoch"]))
        train_loss.append(float(row["train_loss"]))
        val_loss.append(float(row["val_loss"]))
        train_acc.append(float(row["train_acc"]) * 100)
        val_acc.append(float(row["val_acc"]) * 100)


plt.figure()
plt.plot(epochs, train_loss, label="Training Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Learning Curve - Loss")
plt.legend()
plt.grid(True)
plt.savefig(RESULTS_DIR / "loss_curve.png", dpi=150)
plt.show()


plt.figure()
plt.plot(epochs, train_acc, label="Training Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Learning Curve - Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(RESULTS_DIR / "accuracy_curve.png", dpi=150)
plt.show()
