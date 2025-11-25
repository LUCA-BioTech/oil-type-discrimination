import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from .config import *

def load_data():
    df = pd.read_csv(DATA_PATH)

    label_encoder = LabelEncoder()
    df['A_encoded'] = label_encoder.fit_transform(df[LABEL_COLUMN])

    X = df[FEATURE_COLUMNS]
    y = df['A_encoded']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if STRATIFY else None
    )

    return X_train, X_test, y_train, y_test, label_encoder
