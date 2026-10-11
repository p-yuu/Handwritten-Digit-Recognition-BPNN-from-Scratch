from pathlib import Path
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm

from neural_network import NeuralNetwork


BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = BASE_DIR / "data" / "test_images"
MODEL_DIR = BASE_DIR / "outputs" / "models"
MODEL_PATH = MODEL_DIR / "best_model_mnist.npz"      # None：自動選擇使用最新的模型
OUTPUT_PATH = BASE_DIR / "outputs" / "411385019.txt"

DEVICE = "gpu"
BATCH_SIZE = 256


def load_trained_model(model_path, device="gpu"):       # 查看 network 架構並創建相同架構的 network
    with np.load(model_path) as data:
        input_size, hidden_size = data["W1"].shape
        hidden_size_w2, output_size = data["W2"].shape

    if hidden_size != hidden_size_w2:
        raise ValueError("Model weight shapes are inconsistent")

    print(f"Model Architecture: {input_size} -> {hidden_size} -> {output_size}")
    model = NeuralNetwork(input_size=input_size,hidden_size=hidden_size,output_size=output_size,device=device)

    model.load_model(model_path)

    return model

def load_test_images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Test directory not found: {folder}")

    image_paths = sorted(folder.glob("*.png"))
    if not image_paths:
        raise ValueError(f"No PNG images found in {folder}")

    images = []
    for path in tqdm(image_paths, desc="Loading images"):
        with Image.open(path) as img:

            img = img.convert("L")      # 將照片轉為灰階
            if img.size != (28, 28):
                raise ValueError(f"{path.name}: expected 28x28, got {img.size}")

            array = np.asarray(img, dtype=np.float32) / 255.0       # 像素質 [0, 255] -> [0, 1]
            images.append(array.reshape(784))                       # (28,28) -> (784)

    X = np.stack(images, axis=0)

    return X, image_paths   # 回傳影像資料與路徑


def predict_images(model, X, batch_size=256):
    xp = model.xp
    predictions = []

    for start in tqdm(range(0, len(X), batch_size),desc="Recognizing"):
        batch = xp.asarray(X[start:start + batch_size],dtype=xp.float32)

        probabilities = model.forward(batch)        # 只要向前傳播
        labels = xp.argmax(probabilities, axis=1)

        if xp is not np:                    # 如果用 GPU，將預測標籤轉回 CPU NumPy Array
            labels = xp.asnumpy(labels)

        predictions.extend(labels.tolist())

    return predictions


def find_latest_model(model_dir):
    model_dir = Path(model_dir)
    model_paths = list(model_dir.glob("*.npz"))

    if not model_paths:
        raise FileNotFoundError(f"No model files found in {model_dir}")

    return max(model_paths, key=lambda p: p.stat().st_mtime)


def main():
    model_path = MODEL_PATH
    # if MODEL_PATH is None:
    #     model_path = find_latest_model(MODEL_DIR)
    # else:
    #     model_path = MODEL_PATH
    print(f"Using model: {model_path}")

    # 載入模型
    model = load_trained_model(model_path, device="gpu")
    # 載入 test 圖片
    X_test, image_paths = load_test_images(TEST_DIR)

    # 預測
    predictions = predict_images(model,X_test,batch_size=BATCH_SIZE)

    # 寫檔
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for path, digit in zip(image_paths, predictions):
            f.write(f"{path.stem} {digit}\n")

    print(f"\nRecognized {len(predictions)} images.")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()