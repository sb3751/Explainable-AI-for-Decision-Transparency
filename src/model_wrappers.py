# src/model_wrappers.py

import pandas as pd


class PredictProbaWrapper:
    """
    Wraps a sklearn model so predict_proba
    always receives a DataFrame with feature names.
    """

    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.model.predict_proba(X)
