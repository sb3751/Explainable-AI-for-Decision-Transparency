# src/baselines.py

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def train_logistic_regression(X_train, y_train):
    """
    Train an interpretable logistic regression model.
    """
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, max_depth=4):
    """
    Train a shallow decision tree for interpretability.
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
