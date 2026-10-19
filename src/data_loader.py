import struct               # 將二進位資料解讀成整數等 Python 資料類型
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_images(filename, data_dir=DATA_DIR):
    with open(Path(data_dir) / filename, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))

        if magic != 2051:
            raise ValueError("Invalid MNIST image file")

        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)

    return images


def load_labels(filename, data_dir=DATA_DIR):
    with open(Path(data_dir) / filename, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))

        if magic != 2049:
            raise ValueError("Invalid MNIST label file")

        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels


def show_images(images, labels, num_images=10):
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))

    for i, ax in enumerate(axes.flat):
        if i >= min(num_images, 10):
            ax.axis("off")
            continue

        ax.imshow(images[i], cmap="gray")
        ax.set_title(f"Label: {labels[i]}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def one_hot_encode(labels, num_classes=None):
    if num_classes is None:
        if len(labels) == 0:
            raise ValueError("Cannot infer classes from empty labels")
        num_classes = int(np.max(labels)) + 1

    return np.eye(num_classes, dtype=np.float32)[labels]

def preprocess_images(images):
    images = images.reshape(images.shape[0], -1) # 壓成一維向量
    images = images.astype(np.float32) / 255.0   # Pixel Value 轉成 0–1

    return images


if __name__ == "__main__":
    X_train = load_images("train-images.idx3-ubyte")
    y_train = load_labels("train-labels.idx1-ubyte")

    X_test = load_images("t10k-images.idx3-ubyte")
    y_test = load_labels("t10k-labels.idx1-ubyte")

    # show_images(X_train, y_train)

    # 影像前處理
    X_train = preprocess_images(X_train)
    X_test = preprocess_images(X_test)

    # 將 label 轉成 one-hot 向量
    y_train_onehot = one_hot_encode(y_train)
    y_test_onehot = one_hot_encode(y_test) 

    print("Training images:", X_train.shape)
    print("Training labels:", y_train.shape)
    print("Testing images:", X_test.shape)
    print("Testing labels:", y_test.shape)

    print("First label:", y_train[0])
    print("Pixel range:", X_train.min(), X_train.max())

    print("After Flatten:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    print("Normalized range:")
    print(X_train.min(), X_train.max())

    print("Original label:", y_train[0])
    print("One-hot label:", y_train_onehot[0])
    print("One-hot shape:", y_train_onehot.shape)

    print("\n===== Converted Dataset Test =====")

    converted_dir = DATA_DIR / "converted"

    X = load_images("train-images.idx3-ubyte", converted_dir)
    y = load_labels("train-labels.idx1-ubyte", converted_dir)

    X = preprocess_images(X)
    y_onehot = one_hot_encode(y)

    print("Images shape:", X.shape)
    print("Labels shape:", y.shape)
    print("One-hot shape:", y_onehot.shape)
    print("Unique labels:", np.unique(y))
