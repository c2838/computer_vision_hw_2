# 電腦視覺作業二：CNN 正規化與綜合優化

本作業以 MNIST 手寫數字辨識為例，在教師提供的 CNN 上分別實作 Weight Decay、Dropout、Data Augmentation、Early Stopping 與 Batch Normalization，觀察各方法對過擬合與辨識準確率的影響，最後比較多種方法的組合。

學號：`5115056028`。完整程式、訓練輸出、模型架構、曲線與實驗分析均保留於 notebook。

## 檔案

| 檔案 | 用途 |
|---|---|
| `[5115056028]_ch3_課堂練習.ipynb` | Part 0–6 的完整實作與已儲存結果；作業繳交檔案 |
| `model_improvement _課堂實作.pptx` | 教師提供的作業規範 |
| `requirements.txt` | 本機 Python 環境的套件版本清單 |
| `README.md` | 作業概述、執行方式與結果摘要 |

## 執行環境

本次實驗使用 macOS／Apple Silicon、Python 3.12.14 與 CPU 執行。

| 核心套件 | 本次使用版本 | 用途 |
|---|---|---|
| TensorFlow | 2.20.0 | 模型訓練、資料輸入與評估 |
| Keras | 3.15.1 | 建立模型、增強層與 callbacks；程式透過 `tensorflow.keras` 使用 |
| NumPy | 2.5.3 | 陣列、資料處理與隨機抽樣 |
| pandas | 3.0.6 | 整理實驗比較表 |
| matplotlib | 3.11.2 | 繪製圖片預覽與訓練曲線 |
| scikit-learn | 1.9.1 | 範例保留的資料處理套件匯入 |
| ipykernel | 7.4.0 | 提供 notebook 的 Python kernel |

`requirements.txt` 同時包含環境中的相依套件及 JupyterLab。若使用支援 notebook 的 IDE，例如 Antigravity，可直接在 IDE 中執行，不必另外啟動 JupyterLab。本次結果不需要 GPU 或 `tensorflow-metal`。

## 開啟與執行

在作業資料夾開啟終端機。若已有本次使用的 `.venv`，啟動它即可：

```bash
source .venv/bin/activate
```

若是在另一個環境重建，先確認可使用 Python 3.12，再建立虛擬環境並安裝套件：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

接著在 IDE 開啟 `[5115056028]_ch3_課堂練習.ipynb`，將 kernel 選為作業資料夾中的 `.venv/bin/python`。

- **查看既有結果**：直接閱讀 notebook，無須執行任何 cell。
- **重新執行全部實驗**：使用 Restart Kernel & Run All，依 Part 0–6 的順序執行。Part 6 包含 17 次候選實驗，完整重跑會重新訓練所有候選。
- **使用 JupyterLab 介面**：啟動環境後執行 `python -m jupyterlab`；此方式為選用。

MNIST 由 `keras.datasets.mnist.load_data()` 載入；首次執行且尚無快取時需要下載資料。

### Notebook 與模型保存

儲存 `.ipynb` 會保留程式碼、文字說明及當時的輸出，**不會自動保存記憶體中的模型、權重或 History 物件**。目前最佳權重由 callback 暫存在記憶體中，未另外匯出為 `.keras` 或權重檔案。

因此，關閉 kernel 後直接執行最後評估 cell，可能出現 `NameError`，例如找不到 `combined_model_9`。若只是要查看作業結果，保留已儲存的輸出即可；若要再次評估，則須先執行相關資料、函式與模型訓練 cell，重建所需狀態。只重新建立模型架構，仍無法取得先前訓練好的權重。

## 資料與基礎模型

資料集為 MNIST，包含 0–9 共十類、28 × 28 的灰階圖片。

- Training Set：每類 20 張，共 200 張，刻意使用少量資料觀察過擬合。
- Validation Set：從剩餘訓練資料中抽取 5,000 張，與 Training Set 不重疊。
- Test Set：使用 MNIST 原有的 10,000 張測試資料。
- 前處理：轉為 `float32`、像素除以 255，並加入單一通道維度。
- 固定資料切分 seed：42；訓練資料 shuffle seed：2026，每個 epoch 重新洗牌。

基礎 CNN 架構如下；Conv 使用 3 × 3 kernel、`same` padding 與 ReLU，Pooling 使用 2 × 2 最大池化。

```text
Input (28, 28, 1)
→ Conv2D(64) → Conv2D(64) → MaxPooling2D
→ Conv2D(128) → Conv2D(128) → MaxPooling2D
→ Flatten
→ Dense(2048, ReLU)
→ Dense(1024, ReLU)
→ Dense(512, ReLU)
→ Dense(10, Softmax)
```

原始模型共 15,734,218 個參數。各實驗維持原本的 Conv、Pooling 與 Dense 規模，使用固定初始化，基礎訓練設定為 Adam、learning rate 0.001、batch size 32、100 epochs 上限，以及 `sparse_categorical_crossentropy`。

## 各 Part 的實驗設計

| Part | 方法與設定 | 模型／權重選擇依據 |
|---|---|---|
| 0 | 執行原始 baseline，觀察過擬合 | 沿用範例，以訓練完成時的權重評估 |
| 1 | 僅在 Adam 加入 `weight_decay`，比較 0.0001、0.001 | 最高 Validation Accuracy；選擇 0.0001 |
| 2 | 僅在三個隱藏 Dense 的 activation 後加入 Dropout，比較 0.5、0.2 | 最高 Validation Accuracy；選擇 0.5 |
| 3 | 僅使用 RandomTranslation，水平與垂直範圍皆為 ±10% | 最高 Validation Accuracy |
| 4 | 僅使用 Early Stopping，監控 `val_loss`，比較 patience 10、5 | 最低 Validation Loss；同分時選較短訓練，採 patience 5 |
| 5 | 四個 Conv 後加入 BN，再接 ReLU，比較 momentum 0.99、0.9 | 最高 Validation Accuracy；選擇 momentum 0.9 |
| 6 | 比較 Weight Decay、Dropout、Translation、Rotation、Zoom、BN 與 Early Stopping 的組合 | 比較候選的最高 Validation Accuracy；選擇 Trial 9 |

Part 1–5 各自僅使用指定的方法。Part 5 依上課說明，維持原本每兩個 Conv 後才 Pooling 的位置，只將卷積內建的 ReLU 拆出，使順序成為 Conv → BN → ReLU。Keras 的 `fit()` 與 `evaluate()` 分別處理訓練與推論行為。

Data Augmentation 僅在訓練時啟用；Validation 與 Test 保留原本前處理。平移可模擬數字位置差異，小幅旋轉可模擬書寫傾斜；填補使用黑色背景，插值使用 bilinear。Dropout 與 Weight Decay 則用來減少模型對少量樣本及特定特徵的依賴。

除了 Part 0 外，評估前會恢復指定準則下的最佳權重；Part 4 採最低 `val_loss`，其他 Part 採最高 `val_accuracy`。

## 結果摘要

以下數值來自 notebook 已儲存的實際輸出，單位為百分比；Train–Validation Gap 為百分點。

| Part | Train Accuracy | Validation Accuracy | Test Accuracy | Train–Validation Gap |
|---|---:|---:|---:|---:|
| 0：Baseline | 100.00% | 90.56% | 92.09% | 9.44 |
| 1：Weight Decay | 100.00% | 91.06% | 92.90% | 8.94 |
| 2：Dropout | 100.00% | 92.48% | 93.54% | 7.52 |
| 3：Translation | 100.00% | 94.32% | 95.10% | 5.68 |
| 4：Early Stopping | 99.00% | 89.46% | 90.64% | 9.54 |
| 5：Batch Normalization | 100.00% | 89.98% | 91.32% | 10.02 |
| 6：Combined Methods | 99.50% | 95.32% | 95.55% | 4.18 |

Part 0 的最高歷史 Validation Accuracy 為 90.58%，表中的 90.56% 是第 100 個 epoch 權重的實際評估值。其他 Part 的表格值則來自各自選定並還原的權重。

### Part 6 最終模型：Trial 9

| 項目 | 設定／結果 |
|---|---|
| Optimizer | Adam，learning rate = 0.001 |
| Weight decay | 0.0001 |
| Dropout | 三個隱藏 Dense 後各加入 0.2 |
| 增強順序 | Rotation → Translation |
| Rotation | factor = 0.04，即 ±14.4°；seed = 4102 |
| Translation | 水平／垂直 factor = 0.1，即各 ±10%；seed = 4101 |
| 填補／插值 | `constant`、`fill_value=0.0`、`bilinear` |
| Zoom／BN／Early Stopping | 最終模型未使用 |
| Batch size／訓練 epochs | 32／100 |
| 最佳 epoch | 71 |
| Best Validation Accuracy | 95.32% |
| Final Test Accuracy | 95.55% |
| 教師參考 Baseline Test Accuracy | 92.70% |
| Improvement | 95.55% − 92.70% = **2.85 個百分點** |

Part 6 共保留 17 次候選的設定與驗證結果，詳細比較表見 notebook。最後補測的 Trial 17（Dropout 0.1）最佳 Validation Accuracy 為 95.20%，未超過 Trial 9，因此最終選擇維持 Trial 9。

相較本機 baseline，最終模型的 Train–Validation Gap 從 9.44 降為 4.18 個百分點，顯示泛化差距縮小；但後期 Validation Loss 仍會上升，過擬合並未完全消失。BN、較高 Dropout、Zoom 或更大的旋轉幅度，在本次候選設定中也未超過最終模型，不能推論方法疊加一定提高準確率。

## 重現範圍與限制

- 本 README 記錄既有的訓練與評估結果；最終整理後未另外完成一次全新 kernel 的完整 Run All。
- 固定資料、初始化與增強 seed，並啟用 TensorFlow deterministic operations，以控制實驗條件；不同硬體或套件版本仍可能產生不同結果。
- 候選與最佳 epoch 的程式選擇依據為 Validation 指標。不過，本次探索曾在查看 Test 結果後繼續補測，不能將最終 Test 結果宣稱為整個調參過程中從未查看的獨立評估。
- 本機 baseline 的 92.09% 與教師參考值 92.70% 用途不同；作業 Improvement 使用教師參考值計算。
