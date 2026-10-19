from pathlib import Path
import json
import struct

import numpy as np
from PIL import Image
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "data" / "training_data"
OUTPUT_DIR = BASE_DIR / "data" / "converted"

IMAGE_SIZE = (28, 28)

# 'auto', 'folders', or 'txt'
INPUT_MODE = 'auto'
LABELS_FILE = INPUT_DIR / 'labels.txt'

# For another labeled split, set to an existing training mapping JSON.
EXISTING_MAPPING_FILE = None


def read_image(path):
    with Image.open(path) as img:
        img = img.convert('L')
        if img.size != IMAGE_SIZE:
            raise ValueError(f'{path}: expected 28x28, got {img.size}')
        return np.asarray(img, dtype=np.uint8)
    

# folder version
def load_folder_dataset(input_dir):
    label_dirs = sorted((p for p in input_dir.iterdir() if p.is_dir()), key=lambda p: p.name)
    if not label_dirs:
        raise ValueError('No class folders found')
    images, labels = [], []
    for label_dir in label_dirs:
        paths = sorted(label_dir.glob('*.png'))
        if not paths:
            continue
        print(f'Class {label_dir.name}: {len(paths)} images')
        for path in tqdm(paths, desc=f'Loading {label_dir.name}'):
            images.append(read_image(path))
            labels.append(label_dir.name)
    return images, labels


# ground_truth.txt version
def load_txt_dataset(input_dir, labels_file):
    if not labels_file.is_file():
        raise FileNotFoundError(f'Ground truth file not found: {labels_file}')
    image_dir = input_dir / 'images' if (input_dir / 'images').is_dir() else input_dir
    images, labels, seen = [], [], set()
    with labels_file.open('r', encoding='utf-8-sig') as f:
        for line_no, raw_line in enumerate(tqdm(f, desc='Loading TXT labels'), 1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f'{labels_file}:{line_no}: expected "image_id label", got {line!r}')
            image_id, label = parts
            if image_id in seen:
                raise ValueError(f'{labels_file}:{line_no}: duplicate image ID {image_id!r}')
            seen.add(image_id)
            filename = image_id if image_id.lower().endswith('.png') else image_id + '.png'
            path = image_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f'{labels_file}:{line_no}: missing image {path}')
            images.append(read_image(path))
            labels.append(label)
    png_count = sum(1 for _ in image_dir.glob('*.png'))
    if png_count != len(images):
        raise ValueError(f'PNG count ({png_count}) differs from TXT entries ({len(images)})')
    return images, labels


# 將文字標籤轉成數字 (例如 "cat" -> "1") 並存成 JSON 檔
def make_mapping(labels, existing_mapping_file=None):
    if existing_mapping_file is not None:
        with Path(existing_mapping_file).open('r', encoding='utf-8') as f:
            mapping = json.load(f)['label_to_index']
        unknown = sorted(set(labels) - set(mapping))
        if unknown:
            raise ValueError(f'Labels absent from existing training mapping: {unknown}')
    else:
        classes = sorted(set(labels))
        if len(classes) > 256:
            raise ValueError('IDX label format supports at most 256 distinct class indices')
        mapping = {label: i for i, label in enumerate(classes)}
    return mapping


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
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f'Training directory not found: {INPUT_DIR}')
    mode = ('txt' if LABELS_FILE.is_file() else 'folders') if INPUT_MODE == 'auto' else INPUT_MODE
    if mode == 'txt':
        images, labels = load_txt_dataset(INPUT_DIR, LABELS_FILE)
    elif mode == 'folders':
        images, labels = load_folder_dataset(INPUT_DIR)
    else:
        raise ValueError('INPUT_MODE must be auto, txt, or folders')
    if not images:
        raise ValueError('No PNG images found')

    mapping = make_mapping(labels, EXISTING_MAPPING_FILE)
    X = np.stack(images).astype(np.uint8)
    y = np.asarray([mapping[label] for label in labels], dtype=np.uint8)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_idx_images(X, OUTPUT_DIR / 'train-images.idx3-ubyte')
    save_idx_labels(y, OUTPUT_DIR / 'train-labels.idx1-ubyte')
    mapping_path = OUTPUT_DIR / 'label_mapping.json'
    with mapping_path.open('w', encoding='utf-8') as f:
        json.dump({'label_to_index': mapping}, f, indent=2, ensure_ascii=False)
    print(f'Converted {len(y)} images, {len(mapping)} classes')
    print(f'Label mapping: {mapping}')
    print(f'Output: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()