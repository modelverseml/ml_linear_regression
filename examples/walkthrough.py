"""End-to-end linear regression walkthrough.

Reproduces the steps from the old notebook as a single runnable script:

    1. build a synthetic regression dataset
    2. train/test split + feature scaling
    3. feature selection (RFE, then VIF + p-value backward elimination)
    4. fit five models (sklearn, statsmodels, closed-form, GD, SGD)
    5. compare metrics and coefficients side by side
    6. (optional) residual diagnostic plots

Run from the repository root:

    python examples/walkthrough.py            # print the full walkthrough
    python examples/walkthrough.py --plot     # also show residual diagnostics
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from _helpers import add_repo_root_to_path

add_repo_root_to_path()

from linear_regression import (  # noqa: E402  (import after the sys.path tweak)
    ClosedFormRegressionModel,
    GradientDescentRegressionModel,
    RegressionMetrics,
    RfeClass,
    SkLearnRegressionModel,
    SMRegressionModel,
    StochasticGradientDescentRegressionModel,
    VIF,
    final_data,
)

RANDOM_STATE = 42


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def build_dataset():
    """Create a synthetic regression dataset with some uninformative features."""
    X, y, true_coef = make_regression(
        n_samples=500,
        n_features=15,
        n_informative=8,
        noise=15.0,
        coef=True,
        random_state=RANDOM_STATE,
    )
    feature_names = [f"feat_{i}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name="target")

    informative = [f for f, c in zip(feature_names, true_coef) if c != 0]
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Informative features (true coef != 0): {informative}")
    return X, y


def split_and_scale(X, y):
    """Split into train/test and standardise the features."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def select_features(X_train, y_train):
    """Run RFE, inspect VIFs, then backward-eliminate on p-value and VIF."""
    section("Feature selection: Recursive Feature Elimination")
    top_columns = list(RfeClass(X_train, y_train, number_of_features=10).get_rfe_output())
    print(f"RFE-selected columns ({len(top_columns)}): {top_columns}")

    section("Feature selection: Variance Inflation Factor")
    print(VIF(X_train[top_columns]).get_vif_values().to_string(index=False))

    section("Feature selection: backward elimination (p-value + VIF)")
    final_features = final_data(X_train[top_columns], y_train)
    print(f"\nFinal features ({len(final_features)}): {final_features}")
    return final_features


def report_metrics(name, y_true, y_pred, n_features):
    print(f"\n--- {name} ---")
    RegressionMetrics(y_true, y_pred, n_features=n_features).get_metrics()


def fit_models(X_train, X_test, y_train, y_test, features):
    """Fit all five models on the selected features and return them."""
    Xtr, Xte = X_train[features], X_test[features]
    p = len(features)

    section("Model 1: scikit-learn LinearRegression")
    sk = SkLearnRegressionModel(Xtr, y_train)
    sk.build_model()
    print(sk.get_parameters().to_string(index=False))
    report_metrics("Train", y_train, sk.predict(Xtr), p)
    report_metrics("Test", y_test, sk.predict(Xte), p)

    section("Model 2: statsmodels OLS")
    sm_model = SMRegressionModel(Xtr, y_train)
    sm_model.build_model()
    print(sm_model.summary())
    report_metrics("Test", y_test, sm_model.predict(Xte), p)

    section("Model 3: closed-form (normal equation)")
    cf = ClosedFormRegressionModel(Xtr, y_train)
    cf.build_model()
    print(cf.get_parameters().to_string(index=False))
    report_metrics("Test", y_test, cf.predict(Xte), p)

    section("Model 4: batch gradient descent")
    gd = GradientDescentRegressionModel(Xtr, y_train, learning_rate=0.1, n_iterations=10000)
    gd.build_model()
    print(gd.get_parameters().to_string(index=False))
    report_metrics("Test", y_test, gd.predict(Xte), p)

    section("Model 5: mini-batch SGD")
    sgd = StochasticGradientDescentRegressionModel(
        Xtr, y_train, epochs=100, batch_size=32, learning_rate=0.01
    )
    sgd.build_model()
    print(sgd.get_parameters().to_string(index=False))
    report_metrics("Test", y_test, sgd.predict(Xte), p)

    return {"sklearn": sk, "statsmodels": sm_model, "Closed-form": cf,
            "Gradient Desc": gd, "SGD": sgd}


def compare_coefficients(models, features):
    """Line up each model's coefficients in one table."""
    section("Side-by-side coefficient comparison")
    ordered = ["Intercept"] + features

    def coefs(model):
        params = model.get_parameters().set_index("Feature")
        return params.reindex(ordered)["Coefficient"].values

    comparison = pd.DataFrame({"Feature": ordered})
    for name, model in models.items():
        comparison[name] = coefs(model)
    print(comparison.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Run the linear regression walkthrough.")
    parser.add_argument(
        "--plot", action="store_true", help="show residual diagnostic plots at the end"
    )
    args = parser.parse_args()

    # Seed numpy so the SGD shuffling (and the run as a whole) is reproducible.
    np.random.seed(RANDOM_STATE)

    section("Dataset")
    X, y = build_dataset()

    section("Train / test split + scaling")
    X_train, X_test, y_train, y_test = split_and_scale(X, y)

    features = select_features(X_train, y_train)
    models = fit_models(X_train, X_test, y_train, y_test, features)
    compare_coefficients(models, features)

    if args.plot:
        section("Residual diagnostics (scikit-learn model)")
        RegressionMetrics(
            y_test, models["sklearn"].predict(X_test[features]), n_features=len(features)
        ).plot_residuals()

    print()


if __name__ == "__main__":
    main()
