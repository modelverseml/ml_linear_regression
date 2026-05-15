"""
statsmodels OLS Wrapper
-----------------------
Same fitted model as scikit-learn's LinearRegression, but statsmodels also
exposes the per-coefficient standard errors, t-statistics, p-values, and the
overall F-statistic — which is what we use for hypothesis testing on the
coefficients in `automated_feature_selection.py`.
"""

import pandas as pd
import statsmodels.api as sm


class SMRegressionModel:

    def __init__(self, X_train, y_train):

        self.X_train = X_train
        self.y_train = y_train
        self.lr = None

    def build_model(self):
        """Fit OLS via statsmodels with an explicit intercept column."""

        # statsmodels does not add an intercept by default — add_constant prepends 1s.
        X_train_sm = sm.add_constant(self.X_train)
        self.lr = sm.OLS(self.y_train, X_train_sm).fit()

    def predict(self, X):

        return self.lr.predict(sm.add_constant(X))

    def get_parameters(self):
        """Return a DataFrame of (feature, coefficient) pairs."""

        return pd.DataFrame({
            "Feature": self.lr.params.index,
            "Coefficient": self.lr.params.values.round(3),
        })

    def summary(self):
        """Full OLS summary table with std-errors, t/p-values, R², F-stat."""

        return self.lr.summary()
