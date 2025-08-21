# import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('train_data_titanic.csv')

# 觀察資料
df.head()
df.info()
df.describe()

# 資料前處理
# 把Name, Ticket欄位刪除
df.drop(['Name', 'Ticket'], axis=1, inplace=True)

# 用seaborn pairplot來看資料關聯性
sns.pairplot(df[["Survived", "Fare"]], dropna=True)
sns.pairplot(df[["Survived", "PassengerId"]], dropna=True)
sns.pairplot(df[["Survived", "Pclass"]], dropna=True)
sns.pairplot(df[["Survived", "Age"]], dropna=True)
sns.pairplot(df[["Survived", "SibSp"]], dropna=True)
sns.pairplot(df[["Survived", "Parch"]], dropna=True)

# 將df以Survived為主分類，後取得每個欄位的平均值(只選擇數值型態的欄位)
df.groupby('Survived').mean(numeric_only=True)

'''
可以得到以下結論:
(1)生存者的平均票價較高
(2)生存者的平均艙等較高
(3)生存者的平均年齡較低
(4)生存者的平均同輩數較低
(5)生存者的平均父母子女數較高
'''

# 計算特定欄位的數量
df["SibSp"].value_counts()
df["Parch"].value_counts()
df["Sex"].value_counts()

# 計算每個欄位的空值數量
df.isnull().sum()
# 以降冪方式顯示
df.isnull().sum().sort_values(ascending=False)
# 顯示每個欄位空值數量是否超過總數的一半
# 可以用寫條件式的方式來判斷如何處理空值
df.isnull().sum() > len(df)/2
# Cabin欄位空值數量超過總數的一半，因此刪除Cabin欄位
df.drop("Cabin", axis=1, inplace=True)

'''
資料空值常見的填補方式:
(1)平均值
(2)隨機數
(3)最多重複出現的數
(4)複製上一個值
(5)其他
'''

# 填補年齡(Age)欄位的空值
# 性別為主分類，取得個性別的年齡中位數
df.groupby("Sex")["Age"].median()
# 將年齡欄位以個別性別的年齡中位數填補空值
df.groupby("Sex")["Age"].transform("median")
# 用上面的策略去填補Age欄位的空值
df["Age"] = df["Age"].fillna(df.groupby("Sex")["Age"].transform("median"))

# 填補Embarked欄位的空值
# 取得Embarked欄位中各分類的數量，並回傳最大值的分類
df["Embarked"].value_counts().idxmax()
# 用上面的策略去填補Embarked欄位的空值
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].value_counts().idxmax())

# 將資料數值化(用get_dummies)
df = pd.get_dummies(data=df, dtype=int, columns=["Sex", "Embarked"])

# 將女性欄位刪除，以男性欄位的0,1做判斷即可
df.drop("Sex_female", axis=1, inplace=True)

# 若有需要，可以將資料作正規化!! 避免每個欄位級距差異太大

# 觀察各欄位之間的相關性
df.corr()

'''
可以得到以下結論:
(1) Survived與Fare, Pclass, Sex有較高的相關性
(2) Fare與Pclass有較高的相關性，可以考慮刪除其中一個欄位，避免強化影響度
'''

# 刪除Pclass欄位
df.drop("Pclass", axis=1, inplace=True)

# 嘗試刪除欄位優化模型accuracy
# df.drop(["Age","Pclass","SibSp","Parch"], axis=1, inplace=True)
# df.drop(["Embarked_C","Embarked_Q","Embarked_S"], axis=1, inplace=True)

# 準備訓練資料
X = df.drop(["Survived"], axis=1)
y = df["Survived"]

# 切分訓練資料
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=67)

# 準備模型
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=800)
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 評估模型
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
print("Accuracy: ", accuracy_score(y_test, y_pred))
# print("Precision: ", precision_score(y_test, y_pred))
# print("Recall: ", recall_score(y_test, y_pred))

'''
Accuracy: 正確率，=預測正確數/總數=(146+77)/268
FP--Precision: 精確率，=預測存活&&實際也存活數/預測存活數=77/(16+77)
FN--Recall: 召回率，=預測存活&&實際也存活數/實際存活數=77/(29+77)
'''

# 畫出混淆矩陣
pd.DataFrame(confusion_matrix(y_test, y_pred), columns=["Predicted not Survived", "Predicted Survived"], index=["True not Survived", "True Survived"])

# 匯出模型
import joblib
joblib.dump(model, "Titanic_model_export.pkl", compress=3)
