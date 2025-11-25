from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
from .data_loader import load_data
from .models import get_models

def train_evaluate(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred, average='macro'),
        "precision": precision_score(y_test, y_pred, average='macro'),
        "f1": f1_score(y_test, y_pred, average='macro'),
        "kappa": matthews_corrcoef(y_test, y_pred)
    }

    print(f"\n模型：{clf.__class__.__name__}")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return metrics

def main():
    X_train, X_test, y_train, y_test, _ = load_data()

    classifiers, voting_clf = get_models()

    # single models
    for name, clf in classifiers.items():
        train_evaluate(clf, X_train, y_train, X_test, y_test)

    # voting model
    print("\n=== Voting Classifier ===")
    train_evaluate(voting_clf, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
