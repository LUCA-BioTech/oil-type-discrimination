import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from .config import FEATURE_COLUMNS

def predict(model_path, input_csv):
    model = joblib.load(model_path)

    df = pd.read_csv(input_csv)
    X = df[FEATURE_COLUMNS]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    preds = model.predict(X_scaled)

    df['predicted_label'] = preds
    df.to_csv("results/predictions/pred_result.csv", index=False)
    print("Saved to results/predictions/pred_result.csv")
