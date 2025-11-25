from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from catboost import CatBoostClassifier

def get_models():
    classifiers = {
        'DecisionTree': DecisionTreeClassifier(criterion='entropy', random_state=42),
        'KNeighbors': KNeighborsClassifier(weights='distance'),
        'GaussianNB': GaussianNB(var_smoothing=1e-2),
        'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=100,
                             learning_rate_init=0.01, random_state=42),
        'LDA': LinearDiscriminantAnalysis(),
        'CatBoost': CatBoostClassifier(silent=True),
        'ExtraTrees': ExtraTreesClassifier()
    }

    voting_clf = VotingClassifier(
        estimators=[(name, clf) for name, clf in classifiers.items()],
        voting='soft'
    )

    return classifiers, voting_clf
