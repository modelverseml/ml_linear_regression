"""
Recursive Feature Elimination (RFE) Wrapper
-------------------------------------------
Thin wrapper around scikit-learn's RFE. Starting from the full feature set,
RFE fits the estimator, ranks features by their absolute coefficient (or
`feature_importances_` for tree models), drops the weakest, and repeats
until only `n_features_to_select` remain. Use this when you want a fixed
number of features ranked by their contribution to a linear model.
"""

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


class RfeClass:

    def __init__(self, X_train, y_train, number_of_features):

        self.X_train = X_train
        self.y_train = y_train
        self.number_of_features = number_of_features
        self.lm = None
        self.rfe = None
        self.top_columns = None

    def get_rfe_output(self):
        """Fit RFE and store the names of the selected columns."""

        self.lm = LinearRegression().fit(self.X_train, self.y_train)
        self.rfe = RFE(self.lm, n_features_to_select=self.number_of_features)
        self.rfe.fit(self.X_train, self.y_train)

        # rfe.support_ is a boolean mask aligned with X_train.columns.
        self.top_columns = self.X_train.columns[self.rfe.support_]
        return self.top_columns
