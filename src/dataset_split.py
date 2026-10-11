import numpy as np


def train_validation_split(X,y,val_size=5000,seed=42):
    n_samples = len(X)

    if len(y) != n_samples:
        raise ValueError("X and y have different lengths")
    if not 0 < val_size < n_samples:
        raise ValueError("Invalid validation size")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)    # 隨機打亂索引值

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_val = X[val_indices]
    y_val = y[val_indices]

    return X_train, X_val, y_train, y_val
