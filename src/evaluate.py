import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from model.ml_model import MLModel
import pandas as pd

df = pd.read_csv("data/raw/data-923.csv")
X = df[[f"en{i}" for i in range(1,16)]].values
y = df["cate"].values

model = MLModel()
model.fit(X, y)

y_pred = model.predict(X)
print(classification_report(y, y_pred))
cm = confusion_matrix(y, y_pred)
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.savefig("experiments/base_model/confusion_matrix.png")
