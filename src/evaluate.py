import joblib
from sklearn.metrics import classification_report
from .data_loader import load_data

def evaluate_model(model_path):
    X_train, X_test, y_train, y_test, _ = load_data()
    clf = joblib.load(model_path)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

