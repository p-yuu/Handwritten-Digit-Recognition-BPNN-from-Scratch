import csv
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "outputs" / "results"


def find_best_experiment():
    summary_path = RESULTS_DIR / "experiments_summary.csv"

    with open(summary_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        results = list(reader)

    if not results:
        raise ValueError("Experiment summary is empty.")

    best = max(results,key=lambda row: float(row["val_acc"]))

    print(f"Best Experiment: {best['name']}")
    print(f"Validation Accuracy: {float(best['val_acc']) * 100:.2f}%")

    return best["name"]


def load_history(experiment_name):
    history_path = RESULTS_DIR / f"{experiment_name}_history.csv"

    if not history_path.is_file():
        raise FileNotFoundError(f"History file not found: {history_path}")

    epochs = []
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    with open(history_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            train_acc.append(float(row["train_acc"]) * 100)
            val_acc.append(float(row["val_acc"]) * 100)

    if not epochs:
        raise ValueError("Training history is empty.")

    return epochs, train_loss, val_loss, train_acc, val_acc


def plot_learning_curves(experiment_name,epochs,train_loss,val_loss,train_acc,val_acc):
    min_epoch = min(epochs)
    max_epoch = max(epochs)

    tick_step = 1 if max_epoch <= 30 else 5
    epoch_ticks = range(min_epoch, max_epoch + 1, tick_step)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss,marker="o",markersize=3,label="Training Loss")
    plt.plot(epochs, val_loss,marker="o",markersize=3,label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"Learning Curve - Loss ({experiment_name})")

    plt.xlim(min_epoch, max_epoch)
    plt.xticks(epoch_ticks)

    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    loss_path = RESULTS_DIR / f"{experiment_name}_loss_curve.png"
    plt.savefig(loss_path, dpi=200)
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc,marker="o",markersize=3,label="Training Accuracy")
    plt.plot(epochs, val_acc,marker="o",markersize=3,label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Learning Curve - Accuracy ({experiment_name})")

    plt.xlim(min_epoch, max_epoch)
    plt.xticks(epoch_ticks)

    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    accuracy_path = RESULTS_DIR / f"{experiment_name}_accuracy_curve.png"
    plt.savefig(accuracy_path, dpi=200)
    plt.show()
    plt.close()

    print(f"\nSaved: {loss_path}")
    print(f"Saved: {accuracy_path}")


def main():
    best_experiment = find_best_experiment()

    epochs, train_loss, val_loss, train_acc, val_acc = (load_history(best_experiment))

    print(f"History Epochs: {len(epochs)}")
    print(f"Epoch Range: {min(epochs)} - {max(epochs)}")

    plot_learning_curves(best_experiment,epochs,train_loss,val_loss,train_acc,val_acc)


if __name__ == "__main__":
    main()
