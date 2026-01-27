import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    X = df[[f"en{i}" for i in range(1, 16)]].values
    y = df["cate"].values
    return X, y

def scale_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)
