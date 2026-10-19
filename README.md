# Handwritten Digit Recognition — BPNN from Scratch

使用 Python、NumPy／CuPy 實作單隱藏層 Backpropagation Neural Network（BPNN），進行 0–9 手寫數字辨識。訓練包含 Mini-batch SGD、ReLU、Softmax、Cross-Entropy，以及手動計算 Backpropagation 梯度；不使用深度學習框架自動微分。

## 1. 專案結構

```text
project1/
├── data/
│   ├── train-images.idx3-ubyte     # MNIST 原始訓練圖片
│   ├── train-labels.idx1-ubyte     # MNIST 原始訓練標籤
│   ├── t10k-images.idx3-ubyte      # MNIST 原始測試圖片
│   ├── t10k-labels.idx1-ubyte      # MNIST 原始測試標籤
│   ├── training_data/              # 助教提供的分類 PNG（依 0/～9/ 分類）
│   ├── test_images/                # 待預測 PNG
│   └── converted/                  # PNG → IDX 的輸出（執行轉換後）
├── outputs/
│   ├── models/                     # 訓練後的 .npz 模型
│   ├── results/                    # CSV 與訓練曲線
│   └── 411385019.txt               # 最終預測結果
├── src/
│   ├── data_loader.py              # IDX 讀取、正規化與 One-hot Encoding
│   ├── dataset_split.py            # Training / Validation Split
│   ├── neural_network.py           # BPNN、Backpropagation、Checkpoint、模型存取
│   ├── train.py                    # 單一設定訓練
│   ├── experiments.py              # 超參數比較、最佳模型選擇
│   ├── png_to_idx.py               # 分類 PNG → MNIST IDX
│   ├── predict.py                  # PNG 推論與 TXT 輸出
│   ├── check_predictions.py        # 有 Ground Truth 時檢查正確率
│   ├── export_test_png.py          # 匯出 MNIST 測試 PNG（本地驗證）
│   └── plot_results.py             # 畫 Loss / Accuracy 曲線
├── requirements.txt
└── README.md
```


## 2. 環境安裝

使用 Python 3.11 與 Conda 虛擬環境，在 Windows PowerShell 中：

```powershell
conda activate dl_project1
python -m pip install -r requirements.txt
```

`requirements.txt` 包含 NVIDIA CUDA 12.x 適用的 `cupy-cuda12x`。CuPy 需要相容的 NVIDIA 驅動及 CUDA 執行環境；若首次執行 GPU 程式遇到 CUDA 函式庫錯誤，請先確認 GPU／CUDA 設定。**請以 `python -m pip` 安裝，避免使用到 MSYS2 的另一個 Python。**

簡單確認：
```powershell
python -c "import numpy, cupy, PIL, tqdm, matplotlib; print('Dependencies OK')"
```

## 3. 相關設定
- `png_to_idx.py`：
    - 設定 INPUT_MODE 為 'auto', 'folders', or 'txt'

## 4. 模型與訓練流程

目前 BPNN 架構為 `784 → Hidden → 10`：

- Input：28 × 28 灰階圖片，展平成 784 維，像素正規化至 [0, 1]。
- Hidden：ReLU；實驗比較 128、256、512 個神經元。
- Output：10 個類別，Softmax。
- Loss：Cross-Entropy。
- Optimization：手動 Backpropagation + Mini-batch SGD。
- Checkpoint：以 Validation Accuracy 選擇每組實驗的最佳 Epoch，恢復其權重，再比較各組實驗。

### 原始 MNIST

原始 MNIST 共 60,000 張 Training Images 和 10,000 張 Test Images。`experiments.py` 會從 Training Images 切出 Training、Validation（切分比例為 9 : 1），使用 Validation Accuracy 選模型。**MNIST Test Set 不參與超參數選擇**；只在最終模型確定後評估。

### 超參數實驗
六組設定如下：

| Experiment | Learning Rate | Hidden Neurons | Epochs | Batch Size |
|---|---:|---:|---:|---:|
| E01 | 0.01 | 128 | 10 | 64 |
| E02 | 0.05 | 128 | 10 | 64 |
| E03 | 0.10 | 128 | 10 | 64 |
| E04 | 0.05 | 128 | 20 | 64 |
| E05 | 0.05 | 256 | 20 | 64 |
| E06 | 0.05 | 512 | 20 | 64 |

從 `src/` 執行：

```powershell
cd src
python experiments.py
```

結果位於 `outputs/results/`，模型位於 `outputs/models/`。**模型名稱以目前 `experiments.py` 的 `save_model(...)` 實際設定為準**，例如 `best_model.npz` 或 `best_model_ta.npz`。如果已加入 Final Test Evaluation，`final_test_results.csv` 只應在選定模型後產生。