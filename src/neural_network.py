import numpy as np
import cupy as cp
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=128, output_size=10, seed=42, device='gpu'):
        # choose device
        if device == "gpu":
            self.xp = cp
        elif device == "cpu":
            self.xp = np
        else:
            raise ValueError("device must be 'cpu' or 'gpu'")
        xp = self.xp

        rng = np.random.default_rng(seed)

        # 初始化第一層權重
        W1 = rng.standard_normal((input_size, hidden_size)).astype(np.float32) * np.sqrt(2.0 / input_size)
        self.W1 = xp.asarray(W1)
        self.b1 = self.xp.zeros((1, hidden_size),dtype=self.xp.float32)

        # 初始化第二層權重
        W2 = rng.standard_normal((hidden_size, output_size)).astype(np.float32) * np.sqrt(2.0 / hidden_size)
        self.W2 = xp.asarray(W2)
        self.b2 = self.xp.zeros((1, output_size),dtype=self.xp.float32)

    def relu(self, Z):
        return self.xp.maximum(0, Z)

    def softmax(self, Z):
        xp = self.xp
        Z_shifted = Z - xp.max(Z, axis=1, keepdims=True)     # 減去最大值，避免 overflow 影響結果
        exp_Z = xp.exp(Z_shifted)                            # e^x (0 ~ 1)
        return exp_Z / xp.sum(exp_Z, axis=1, keepdims=True)  # 轉成機率

    def forward(self, X):        
        # Hidden Layer
        self.Z1 = X @ self.W1 + self.b1                      # @：Python 矩陣乘法運算子，等同於 np.matmul(X, self.W1)
        self.A1 = self.relu(self.Z1)

        # Output Layer
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.softmax(self.Z2)

        return self.A2

    def compute_loss(self, y_pred, y_true):
        xp = self.xp
        y_pred = xp.clip(y_pred, 1e-12, 1.0)  # 限定範圍在 1e-12 ~ 1.0，避免 log(0)
        
        return -xp.mean(xp.sum(y_true * xp.log(y_pred), axis=1))    # Cross-Entropy Loss
                                                                    # axis：把每張圖片的 10 個類別計算結果加起來

    def relu_derivative(self, Z):           # ReLU 導函數
        return (Z > 0).astype(self.xp.float32)

    def backward(self, X, y_true):
        xp = self.xp
        m = X.shape[0]  # batch size

        # Output layer
        dZ2 = (self.A2 - y_true) / m
        self.dW2 = self.A1.T @ dZ2
        self.db2 = xp.sum(dZ2, axis=0, keepdims=True)

        # Hidden Layer
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        self.dW1 = X.T @ dZ1
        self.db1 = xp.sum(dZ1, axis=0, keepdims=True)

    def update_parameters(self, learning_rate=0.01):    # 更新參數
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1

        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2

    def fit(self,X_train,y_train,epochs=10,batch_size=64,learning_rate=0.01,X_val=None,y_val=None,verbose=True):
        xp = self.xp

        X_train = xp.asarray(X_train, dtype=xp.float32)     # images
        y_train = xp.asarray(y_train, dtype=xp.float32)     # labels

        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be provided together")

        if X_val is not None:
            X_val = xp.asarray(X_val, dtype=xp.float32)
            y_val = xp.asarray(y_val, dtype=xp.float32)

        n_samples = len(X_train)
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        
        best_val_acc = -1.0
        best_epoch = None
        best_weights = None

        for epoch in range(epochs):
            indices = xp.random.permutation(n_samples)      # 隨機打亂 index

            progress = tqdm(
                range(0, n_samples, batch_size),
                desc=f"Epoch {epoch + 1}/{epochs}",
                disable=not verbose
            )

            for start in progress:
                batch_indices = indices[start:start + batch_size]

                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                self.forward(X_batch)
                self.backward(X_batch, y_batch)
                self.update_parameters(learning_rate)

            # Evaluate after all weight updates
            train_loss, train_acc = self.evaluate_metrics(X_train, y_train)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            if X_val is not None:
                val_loss, val_acc = self.evaluate_metrics(X_val, y_val)

                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                if verbose:
                    tqdm.write(
                        f"Epoch {epoch + 1}/{epochs} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Train Acc: {train_acc * 100:.2f}% | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val Acc: {val_acc * 100:.2f}%"
                    )

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch + 1
                    best_weights = {"W1": self.W1.copy(),"b1": self.b1.copy(),"W2": self.W2.copy(),"b2": self.b2.copy()}

            elif verbose:
                tqdm.write(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")

        if best_weights is not None:
            self.W1 = best_weights["W1"]
            self.b1 = best_weights["b1"]
            self.W2 = best_weights["W2"]
            self.b2 = best_weights["b2"]

            history["best_epoch"] = best_epoch
            history["best_val_acc"] = best_val_acc

            if verbose:
                print(f"Best Epoch: {best_epoch} | Best Validation Accuracy: {best_val_acc * 100:.2f}%")

        return history

    def evaluate_metrics(self, X, y_true):
        xp = self.xp

        X = xp.asarray(X, dtype=xp.float32)
        y_true = xp.asarray(y_true, dtype=xp.float32)

        y_pred = self.forward(X)
        loss = self.compute_loss(y_pred, y_true)

        predicted = xp.argmax(y_pred, axis=1)
        actual = xp.argmax(y_true, axis=1)
        accuracy = xp.mean(predicted == actual)

        return float(loss), float(accuracy)

    def save_model(self, filepath=None):
        if filepath is None:
            filepath = Path(__file__).resolve().parent.parent / "models"
        filepath = Path(filepath)
        if filepath.suffix.lower() != ".npz":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = filepath / f"model_{timestamp}.npz"

        filepath.parent.mkdir(parents=True, exist_ok=True)
        xp = self.xp

        # Convert GPU arrays to CPU NumPy arrays
        def to_numpy(array):
            return xp.asnumpy(array) if xp is not np else array
        
        np.savez(filepath,W1=to_numpy(self.W1),b1=to_numpy(self.b1),W2=to_numpy(self.W2),b2=to_numpy(self.b2))

        print(f"Model saved to: {filepath}")
        return filepath


    def load_model(self, filepath):
        with np.load(filepath) as data:
            self.W1 = self.xp.asarray(data["W1"])
            self.b1 = self.xp.asarray(data["b1"])
            self.W2 = self.xp.asarray(data["W2"])
            self.b2 = self.xp.asarray(data["b2"])

        print(f"Model loaded from: {filepath}")


if __name__ == "__main__":
    model = NeuralNetwork()

    print("W1:", model.W1.shape)
    print("b1:", model.b1.shape)
    print("W2:", model.W2.shape)
    print("b2:", model.b2.shape)
