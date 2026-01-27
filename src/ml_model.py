import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

class MLModel:
    def __init__(self):
        self.pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis()),
            ("nb", GaussianNB())
        ])
        self.le = LabelEncoder()

    def fit(self, X, y):
        y_enc = self.le.fit_transform(y.astype(str))
        return self.pipe.fit(X, y_enc)

    def predict(self, X):
        return self.pipe.predict(X)

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)

    def get_classes(self):
        return self.le.classes_
