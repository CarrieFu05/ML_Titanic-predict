# 🛳️ Titanic Survival Prediction (Kaggle)

本專案針對 **Kaggle 經典題目 [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)** 進行資料分析與機器學習建模。  
目標是根據乘客特徵（如艙等、票價、性別、年齡等），預測該乘客是否能在鐵達尼號沉船事故中存活。  

👉 本專案中我嘗試最好的 Kaggle 公開排行榜分數為 **0.75358**。

---

## 📂 專案結構

.
├── train_data_titanic.csv # 訓練資料
├── titanic_model.py # 建模程式碼 (Logistic Regression)
├── Titanic_LR_20241116_2.pkl # 訓練完成的模型 (joblib)
└── README.md # 專案說明


---

## ⚙️ 專案流程

1. **資料探索 (EDA)**  
   - 使用 `pandas` 與 `seaborn` 進行初步探索  
   - 檢查欄位分佈、空值比例、相關性  
   - 視覺化 (pairplot, heatmap)

2. **資料前處理**
   - 刪除高比例缺失值的 `Cabin` 欄位  
   - 以 **性別分組的年齡中位數** 填補 `Age` 缺失值  
   - 以 **出現次數最多的港口** 填補 `Embarked` 缺失值  
   - One-Hot Encoding (`Sex`, `Embarked`)  
   - 移除多餘欄位 (e.g. `Name`, `Ticket`)  
   - 考慮移除高相關性特徵 (`Pclass` vs `Fare`)

3. **模型建立**
   - 使用 `Logistic Regression` 建模 (`sklearn`)  
   - 訓練集與測試集切分 (70% / 30%)  
   - 模型儲存 (`joblib`)

4. **模型評估**
   - 評估指標：`Accuracy`, `Precision`, `Recall`, `Confusion Matrix`  
   - **最佳 Kaggle Score: 0.75358**

---

## 📊 模型表現

- **Accuracy (本地測試集)**: 約 0.82  
- **Precision / Recall**: 使用混淆矩陣計算  
- **Kaggle Public Score**: **0.75358**

| 指標 | 說明 |
|------|------|
| Accuracy | 正確率 = (TP+TN)/(總數) |
| Precision | 精確率 = TP/(TP+FP) |
| Recall | 召回率 = TP/(TP+FN) |

---

## 🖼️ 視覺化成果

### 📌 生存與票價的關係
![fare_vs_survival]()

### 📌 混淆矩陣
|                | 預測未存活 | 預測存活 |
|----------------|------------|----------|
| **實際未存活** | 146        | 16       |
| **實際存活**   | 29         | 77       |

---

## 🚀 如何使用

### 1️⃣ 環境安裝
```bash
pip install -r requirements.txt
```

### 2️⃣ 執行訓練
```bash
python titanic_model.py
```

### 3️⃣ 載入模型並預測
```python
import joblib
import pandas as pd

# 載入模型
model = joblib.load("Titanic_LR_20241116_2.pkl")

# 測試資料 (假設已完成前處理)
X_test = pd.read_csv("test_processed.csv")

# 預測
y_pred = model.predict(X_test)
```

---

## 🔮 未來優化方向
-嘗試 隨機森林 (Random Forest)、梯度提升 (XGBoost / LightGBM) 等模型
-特徵工程：新增 FamilySize, Title (從姓名萃取)

---

## 🧑‍💻 技術棧
-Python (pandas, numpy, matplotlib, seaborn, scikit-learn)
-Joblib (模型儲存)
-Kaggle (比賽平台)

