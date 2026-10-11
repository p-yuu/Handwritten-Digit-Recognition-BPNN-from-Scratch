import csv
import time
from pathlib import Path
import numpy as np

from neural_network import NeuralNetwork
from dataset_split import train_validation_split
from data_loader import load_images,load_labels,preprocess_images,one_hot_encode


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "outputs" / "results"
MODELS_DIR = BASE_DIR / "outputs" / "models"

DATASET = "ta"  # "mnist" or "ta"
DATA_DIRS = {
    "mnist": BASE_DIR / "data",
    "ta": BASE_DIR / "data" / "converted"
}
DATA_DIR = DATA_DIRS[DATASET]

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Experiment configurations
experiments = [
    {"name": "E01", "lr": 0.01, "hidden": 128, "epochs": 10, "batch": 64},
    {"name": "E02", "lr": 0.05, "hidden": 128, "epochs": 10, "batch": 64},
    {"name": "E03", "lr": 0.10, "hidden": 128, "epochs": 10, "batch": 64},
    {"name": "E04", "lr": 0.05, "hidden": 128, "epochs": 20, "batch": 64},
    {"name": "E05", "lr": 0.05, "hidden": 256, "epochs": 20, "batch": 64},
    {"name": "E06", "lr": 0.05, "hidden": 512, "epochs": 20, "batch": 64},
]


def main():
    X = load_images(DATA_DIR / "train-images.idx3-ubyte")
    y = load_labels(DATA_DIR / "train-labels.idx1-ubyte")

    X = preprocess_images(X)
    y = one_hot_encode(y)

    # 拆分 train, val
    val_size = max(1, int(len(X) * 0.1))
    X_train, X_val, y_train, y_val = train_validation_split(X, y, val_size=val_size, seed=42)

    print(f"\nDataset: {DATASET}")
    print(f"Total samples: {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    results = []
    best_accuracy = -1.0
    best_experiment = None

    # 開始實驗
    for config in experiments:
        print(
            f"\n===== {config['name']} =====\n"
            f"LR={config['lr']} | Hidden={config['hidden']} | Epochs={config['epochs']} | Batch={config['batch']}"
        )

        model = NeuralNetwork(input_size=784,hidden_size=config["hidden"],output_size=10,seed=42,device="gpu")
        np.random.seed(42)
        model.xp.random.seed(42)

        start_time = time.perf_counter()

        history = model.fit(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=config["epochs"],
            batch_size=config["batch"],
            learning_rate=config["lr"],
            verbose=True
        )


        model.xp.cuda.Stream.null.synchronize()

        training_time = time.perf_counter() - start_time
        val_loss, val_acc = model.evaluate_metrics(X_val, y_val)

        result = {
            "name": config["name"],
            "lr": config["lr"],
            "hidden": config["hidden"],
            "epochs": config["epochs"],
            "batch": config["batch"],
            "best_epoch": history["best_epoch"],
            "val_loss": val_loss,
            "val_acc": val_acc,
            "training_time": training_time
        }

        results.append(result)
        history_path = RESULTS_DIR / f"{config['name']}_history.csv"
        print(f"Best Epoch: {history['best_epoch']} | Validation Accuracy: {val_acc * 100:.2f}% | Time: {training_time:.2f}s")

        with open(history_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

            for i in range(len(history["train_loss"])):
                writer.writerow([i + 1,history["train_loss"][i],history["train_acc"][i],history["val_loss"][i],history["val_acc"][i]])

        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_experiment = config["name"]

            model.save_model(MODELS_DIR / f"best_model_{DATASET}.npz")

    # Save experiment summary
    summary_path = RESULTS_DIR / "experiments_summary.csv"

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name","lr","hidden","epochs","batch","best_epoch","val_loss","val_acc","training_time"]
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n===== EXPERIMENT SUMMARY =====")

    for result in sorted(results,key=lambda r: r["val_acc"],reverse=True):
        print(f"{result['name']} | Val Acc: {result['val_acc'] * 100:.2f}% | Time: {result['training_time']:.2f}s")

    print(f"\nBest Experiment: {best_experiment}")
    print(f"Best Validation Accuracy: {best_accuracy * 100:.2f}%")
    print(f"Results saved to: {summary_path}")


    # test dataset 評估
    print("\n===== FINAL TEST EVALUATION =====")
    X_test = load_images(DATA_DIR / "t10k-images.idx3-ubyte")
    y_test = load_labels(DATA_DIR / "t10k-labels.idx1-ubyte")

    X_test = preprocess_images(X_test)
    y_test = one_hot_encode(y_test)

    best_model_path = MODELS_DIR / "best_model.npz"

    with np.load(best_model_path) as data:
        input_size, hidden_size = data["W1"].shape
        _, output_size = data["W2"].shape

    best_model = NeuralNetwork(input_size=input_size,hidden_size=hidden_size,output_size=output_size,device="gpu")
    best_model.load_model(best_model_path)

    test_loss, test_acc = best_model.evaluate_metrics(X_test,y_test)
    print(f"Best Experiment: {best_experiment}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    test_result_path = RESULTS_DIR / "final_test_results.csv"
    with open(test_result_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["best_experiment","validation_accuracy","test_loss","test_accuracy"])
        writer.writerow([best_experiment,best_accuracy,test_loss,test_acc])
    print(f"Final test results saved to: {test_result_path}")

if __name__ == "__main__":
    main()
