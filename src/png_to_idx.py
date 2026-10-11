from pathlib import Path
import struct

import numpy as np
from PIL import Image
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "data" / "training_data"
OUTPUT_DIR = BASE_DIR / "data" / "converted"

IMAGE_SIZE = (28, 28)


def load_png_dataset(input_dir):
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {input_dir}")

    images = []
    labels = []

    # Find label folders: 0, 1, ..., 9
    label_dirs = sorted(
        [
            path for path in input_dir.iterdir()
            if path.is_dir() and path.name in {
                str(i) for i in range(10)
            }
        ], key=lambda path: int(path.name)
    )

    if not label_dirs:
        raise ValueError("No label folders (0-9) found")

    for label_dir in label_dirs:
        label = int(label_dir.name)
        image_paths = sorted(label_dir.glob("*.png"))

        print(f"Label {label}: {len(image_paths)} images")

        for path in tqdm(image_paths,desc=f"Loading label {label}"):
            with Image.open(path) as img:
                img = img.convert("L")

                if img.size != IMAGE_SIZE:
                    raise ValueError(f"{path}: expected 28x28, got {img.size}")

                image = np.asarray(img,dtype=np.uint8)
                images.append(image)
                labels.append(label)

    if not images:
        raise ValueError("No PNG images found")

    X = np.stack(images, axis=0)
    y = np.asarray(labels, dtype=np.uint8)

    return X, y


def save_idx_images(images, filepath):
    n, rows, cols = images.shape

    with open(filepath, "wb") as f:
        f.write(struct.pack(">IIII", 2051, n, rows, cols))
        f.write(images.tobytes())


def save_idx_labels(labels, filepath):
    with open(filepath, "wb") as f:
        f.write(struct.pack(">II", 2049, len(labels)))
        f.write(labels.tobytes())


def main():
    print("===== PNG TO IDX CONVERTER =====")

    X, y = load_png_dataset(INPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

    images_path = OUTPUT_DIR / "train-images.idx3-ubyte"
    labels_path = OUTPUT_DIR / "train-labels.idx1-ubyte"

    save_idx_images(X, images_path)
    save_idx_labels(y, labels_path)

    print("\n===== CONVERSION COMPLETED =====")
    print(f"Images shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Images saved to: {images_path}")
    print(f"Labels saved to: {labels_path}")


if __name__ == "__main__":
    main()
