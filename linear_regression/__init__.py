"""Linear regression toolkit.

Five interchangeable ways to fit an ordinary least squares model (two library
wrappers and three from-scratch implementations), plus the feature-selection
and diagnostic tools used around them. Every model exposes the same small
interface: build_model(), predict(X) and get_parameters().

See README.md for the theory and examples/ for a runnable end-to-end walkthrough.
"""

from .automated_feature_selection import final_data
from .closed_form_model_building import ClosedFormRegressionModel
from .gradient_descent_model_building import GradientDescentRegressionModel
from .recursive_feature_elimination import RfeClass
from .regression_metrics import RegressionMetrics
from .sklearn_model_building import SkLearnRegressionModel
from .statsmodel_model_building import SMRegressionModel
from .stochastic_gradient_descent_model_building import (
    StochasticGradientDescentRegressionModel,
)
from .variance_inflation_factor_data import VIF

__all__ = [
    "ClosedFormRegressionModel",
    "GradientDescentRegressionModel",
    "StochasticGradientDescentRegressionModel",
    "SkLearnRegressionModel",
    "SMRegressionModel",
    "RegressionMetrics",
    "RfeClass",
    "VIF",
    "final_data",
]

__version__ = "1.0.0"
