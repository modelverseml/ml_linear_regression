"""Mini-batch stochastic gradient descent linear regression.

Same loss as batch gradient descent, but the gradient is estimated from a small
random mini-batch each step instead of the full dataset. Each update is cheaper
and the added noise can help escape poor regions, at the cost of a noisier
convergence path.

    for each epoch:
        shuffle rows
        for each mini-batch of size batch_size:
            beta := beta - alpha * (1/m) * X_batch^T (X_batch beta - y_batch)
"""

import numpy as np
import pandas as pd


class StochasticGradientDescentRegressionModel:
    def __init__(self, X_train, y_train, epochs, batch_size, learning_rate=0.01):
        self.X_train = X_train
        self.y_train = y_train
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.feature_names = list(X_train.columns)
        self.coefficients = None

    def build_model(self):
        """Fit the model with mini-batch SGD."""
        X = self.X_train.to_numpy()
        Y = self.y_train.to_numpy()

        # Prepend a column of 1s for the intercept term.
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        self.feature_names = ["Intercept"] + self.feature_names

        n_samples = X.shape[0]
        beta = np.zeros(X.shape[1])

        for _ in range(self.epochs):
            # Reshuffle each epoch to decorrelate consecutive mini-batches.
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]

                gradient = X_batch.T @ (X_batch @ beta - Y_batch) / X_batch.shape[0]
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
