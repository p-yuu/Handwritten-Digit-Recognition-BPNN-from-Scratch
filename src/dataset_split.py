import numpy as np


def train_validation_split(X,y,val_ratio=0.2,seed=42):
    n_samples = len(X)

    if len(y) != n_samples:
        raise ValueError("X and y have different lengths")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")

    val_size = round(n_samples * val_ratio)

    if not 0 < val_size < n_samples:
        raise ValueError("Dataset is too small for the given val_ratio")

    rng = np.random.default_rng(seed)   # 產生隨機數
    classes, counts = np.unique(y, return_counts=True)  # 找出有多少不同的 class 和各 class 的數量

    train_indices = []
    val_indices = []

    for cls, count in zip(classes, counts):
        if count < 2:
            raise ValueError(f"Class {cls} has only {count} sample(s). At least 2 are required.")


        cls_indices = np.where(y == cls)[0]     # 找出目前類別的所有圖片 idx        
        rng.shuffle(cls_indices)                # 打亂這個類別的圖片順序
        n_val = round(count * val_ratio)        # 計算這個類別的 Validation 數量
        n_val = max(1, min(n_val, count - 1))

        # 分配 Training、Val
        val_indices.extend(cls_indices[:n_val])
        train_indices.extend(cls_indices[n_val:])

    # 再次打亂
    train_indices = rng.permutation(train_indices)
    val_indices = rng.permutation(val_indices)

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_val = X[val_indices]
    y_val = y[val_indices]

    return X_train, X_val, y_train, y_val
