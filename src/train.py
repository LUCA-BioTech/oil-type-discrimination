import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from model.ml_model import MLModel

# 加载数据
df = pd.read_csv("data/raw/data-923.csv")
X = df[[f"en{i}" for i in range(1,16)]].values
y = df["cate"].values

model = MLModel()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 交叉验证
scores = cross_val_score(model.pipe, X, y, cv=cv, scoring="accuracy")

results = {
    "cv_scores": scores.tolist(),
    "mean_acc": float(np.mean(scores)),
    "std_acc": float(np.std(scores))
}

print(results)

# 保存实验结果到本地
with open("experiments/base_model/metrics.json", "w") as fp:
    json.dump(results, fp, indent=4)
