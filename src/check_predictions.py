from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def read_results(path):
    results = {}

    with open(path, "r") as f:
        for line in f:
            filename, digit = line.split()
            results[filename] = int(digit)

    return results

ground_truth = read_results(BASE_DIR / "data" / "test_images" / "ground_truth.txt")
predictions = read_results(BASE_DIR / "outputs" / "411385019.txt")

correct = 0

for filename, actual in ground_truth.items():
    predicted = predictions.get(filename)

    if predicted == actual:
        correct += 1
    else:
        print(
            f"Wrong: {filename}.png | "
            f"Actual: {actual} | "
            f"Predicted: {predicted}"
        )

accuracy = correct / len(ground_truth)

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print(f"Correct: {correct}/{len(ground_truth)}")
