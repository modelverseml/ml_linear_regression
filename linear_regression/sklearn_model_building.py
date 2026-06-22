"""scikit-learn linear regression wrapper.

Thin wrapper around sklearn.linear_model.LinearRegression that keeps the same
build_model / predict / get_parameters interface as the from-scratch models in
this package, so they can be swapped freely in the walkthrough.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression


class SkLearnRegressionModel:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.lr = None

    def build_model(self):
        """Fit an sklearn LinearRegression on the training data."""
        self.lr = LinearRegression().fit(self.X_train, self.y_train)

    def predict(self, X):
        """Predict y for new inputs X."""
        return self.lr.predict(X)

    def get_parameters(self):
        """Return a DataFrame of (feature, coefficient) pairs, intercept last."""
        coef_df = pd.DataFrame({
            "Feature": self.X_train.columns,
            "Coefficient": self.lr.coef_.round(3),
        })
        coef_df.loc[len(coef_df)] = ["Intercept", round(self.lr.intercept_, 3)]
        return coef_df
