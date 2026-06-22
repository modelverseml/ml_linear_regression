"""Closed-form (normal equation) linear regression.

Solves the OLS problem analytically:

    beta = (X^T X)^(-1) X^T y

This has a unique solution whenever X^T X is invertible. When it is not (for
example with perfectly collinear features) we fall back to the Moore-Penrose
pseudo-inverse, which gives the minimum-norm least-squares solution.
"""

import numpy as np
import pandas as pd


class ClosedFormRegressionModel:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = list(X_train.columns)
        self.coefficients = None

    def build_model(self):
        """Fit the model by solving the normal equation."""
        X = self.X_train.to_numpy()
        Y = self.y_train.to_numpy()

        # Prepend a column of 1s so the first coefficient becomes the intercept.
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        self.feature_names = ["Intercept"] + self.feature_names

        # Use the pseudo-inverse when X^T X is singular (collinear features).
        try:
            gram_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            gram_inv = np.linalg.pinv(X.T @ X)

        self.coefficients = gram_inv @ (X.T @ Y)

    def get_parameters(self):
        """Return a DataFrame of (feature, coefficient) pairs."""
        return pd.DataFrame({
            "Feature": self.feature_names,
            "Coefficient": self.coefficients.round(3),
        })

    def predict(self, X):
        """Predict y for new inputs X (same feature order as training)."""
        X = np.asarray(X)
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.coefficients
