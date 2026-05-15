"""
Batch Gradient Descent Linear Regression
----------------------------------------
Iteratively minimises the mean squared error using the update rule

    beta := beta - alpha * (1/n) * X^T (X beta - y)

`alpha` is the learning rate and `n_iterations` is the number of full
passes over the training set. Unlike the closed-form solution this scales
to very large feature matrices that cannot be inverted directly.
"""

import numpy as np
import pandas as pd


class GradientDescentRegressionModel:

    def __init__(self, X_train, y_train, learning_rate=0.1, n_iterations=10000):

        self.X_train = X_train
        self.y_train = y_train
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.feature_names = list(X_train.columns)
        self.coefficients = None

    def build_model(self):
        """Fit the model with batch gradient descent."""

        X = self.X_train.to_numpy()
        Y = self.y_train.to_numpy()

        # Prepend a column of 1s for the intercept term.
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        self.feature_names = ["Intercept"] + self.feature_names

        n_samples = X.shape[0]
        beta = np.zeros(X.shape[1])

        for _ in range(self.n_iterations):
            # Gradient of (1/n) * sum((Xb - y)^2) with respect to beta.
            gradient = X.T @ (X @ beta - Y) / n_samples
            beta = beta - self.learning_rate * gradient

        self.coefficients = beta

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
