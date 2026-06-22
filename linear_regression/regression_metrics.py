"""Regression metrics and residual diagnostics.

Reports the standard regression metrics (MAE, MSE, RMSE, R-squared, adjusted
R-squared) and draws the residual plots used to check the linear-regression
assumptions discussed in the README:

    - residuals vs fitted  -> linearity and constant variance (homoscedasticity)
    - residual histogram   -> normality of the errors

Adjusted R-squared needs the number of predictors, so pass n_features to enable it.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class RegressionMetrics:
    def __init__(self, actual, predict, n_features=None):
        # Convert to numpy to avoid pandas index-alignment surprises on subtraction.
        self.actual = np.asarray(actual)
        self.predict = np.asarray(predict)
        self.n_features = n_features

    def get_metrics(self):
        """Print MAE, MSE, RMSE, R-squared, and adjusted R-squared if possible."""
        mae = mean_absolute_error(self.actual, self.predict)
        mse = mean_squared_error(self.actual, self.predict)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.actual, self.predict)

        print(f"MAE  : {mae:.4f}")
        print(f"MSE  : {mse:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"R2   : {r2:.4f}")

        if self.n_features is not None:
            # Adjusted R-squared penalises features that do not improve the fit.
            n = len(self.actual)
            p = self.n_features
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            print(f"Adj R2 : {adj_r2:.4f}")

    def plot_residuals(self):
        """Show residuals-vs-fitted and residual-distribution diagnostics."""
        residuals = self.actual - self.predict

        plt.scatter(self.predict, residuals, alpha=0.5)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Fitted")
        plt.show()

        sns.histplot(residuals, kde=True)
        plt.title("Residuals Distribution")
        plt.xlabel("Residual")
        plt.show()
