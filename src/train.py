import time
import numpy as np
from pathlib import Path

from neural_network import NeuralNetwork
from data_loader import load_images,load_labels,preprocess_images,one_hot_encode
from dataset_split import train_validation_split

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Load MNIST
X_train = load_images(DATA_DIR / "train-images.idx3-ubyte")
y_train = load_labels(DATA_DIR / "train-labels.idx1-ubyte")

X_test = load_images(DATA_DIR / "t10k-images.idx3-ubyte")
y_test = load_labels(DATA_DIR / "t10k-labels.idx1-ubyte")

# Preprocessing
X_train = preprocess_images(X_train)
X_test = preprocess_images(X_test)

y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)

X_train, X_val, y_train, y_val = train_validation_split(X_train,y_train,val_size=5000,seed=42)

# Initialize model
model = NeuralNetwork(input_size=784,hidden_size=128,output_size=10,device="gpu")

# Training
start_time = time.perf_counter()
model.fit(X_train,y_train,epochs=10,batch_size=64,learning_rate=0.01,X_val=X_val,y_val=y_val)

# Synchronize GPU before measuring time
model.xp.cuda.Stream.null.synchronize()
training_time = time.perf_counter() - start_time

# Evaluation
test_loss, test_accuracy = model.evaluate_metrics(X_test, y_test)

# Save
model.save_model()

print(f"\nTraining Time: {training_time:.2f} seconds")
print(f"Test Loss: {test_loss * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
