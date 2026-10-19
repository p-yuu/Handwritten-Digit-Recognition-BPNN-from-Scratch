from pathlib import Path
import struct

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "data" / "test_images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Read original MNIST test images
with open(DATA_DIR / "t10k-images.idx3-ubyte", "rb") as f:
    magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
    if magic != 2051:
        raise ValueError("Invalid MNIST image file")
    images = np.frombuffer(f.read(), dtype=np.uint8)
    images = images.reshape(count, rows, cols)

# Read original MNIST test labels
with open(DATA_DIR / "t10k-labels.idx1-ubyte", "rb") as f:
    magic, count = struct.unpack(">II", f.read(8))
    if magic != 2049:
        raise ValueError("Invalid MNIST label file")
    labels = np.frombuffer(f.read(), dtype=np.uint8)

# Export first 100 images
num_images = 10000

with open(OUTPUT_DIR / "ground_truth.txt", "w") as f:
    for i in range(num_images):
        filename = f"{i:04d}.png"

        Image.fromarray(images[i], mode="L").save(OUTPUT_DIR / filename)

        f.write(f"{i:04d} {labels[i]}\n")

print(f"Exported {num_images} PNG images to {OUTPUT_DIR}")
