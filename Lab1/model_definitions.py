"""Model definitions for the MLSM Lab 1 datasets.

Each dataset has its own pipelines and parameter grids so they can be tuned
independently.
"""

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


def _genreall_models(n_features):
    """Models for GenreAll, including variance and feature selection."""
    selector_options = [
        k for k in [80, 100, 130, 150, 180, 200] if k <= n_features
    ]
    selector_options.append("all")

    common = [
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.0)),
    ]

    return {
        "Logistic Regression": (
            Pipeline(common + [
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(score_func=f_classif)),
                ("model", LogisticRegression(max_iter=5000, solver="lbfgs")),
            ]),
            {
                "selector__k": selector_options,
                "model__C": [0.03, 0.1, 0.2, 0.3, 0.5, 1.0],
            },
        ),
        "k-NN": (
            Pipeline(common + [
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(score_func=f_classif)),
                ("model", KNeighborsClassifier()),
            ]),
            {
                "selector__k": selector_options,
                "model__n_neighbors": [5, 7, 9, 11, 15],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        ),
        "Decision Tree": (
            Pipeline(common + [
                ("selector", SelectKBest(score_func=f_classif)),
                ("model", DecisionTreeClassifier(random_state=42)),
            ]),
            {
                "selector__k": selector_options,
                "model__max_depth": [5, 10, 15],
                "model__min_samples_leaf": [5, 10],
                "model__min_samples_split": [2, 10],
            },
        ),
        "Artificial Neural Network": (
            Pipeline(common + [
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(score_func=f_classif)),
                ("model", MLPClassifier(
                    solver="adam",
                    learning_rate_init=0.001,
                    max_iter=700,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=15,
                    tol=1e-3,
                )),
            ]),
            {
                "selector__k": selector_options,
                "model__hidden_layer_sizes": [(50,), (100,), (100, 50)],
                "model__alpha": [0.001, 0.01, 0.1],
            },
        ),
    }


def _features_30_sec_models():
    """Models and parameter grids specifically for the 30-second dataset."""
    return {
        "Logistic Regression": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, solver="lbfgs")),
            ]),
            {"model__C": [0.01, 0.03, 0.1, 0.3, 1.0]},
        ),
        "k-NN": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier()),
            ]),
            {
                "model__n_neighbors": [3, 5, 7, 9, 13, 17],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        ),
        "Decision Tree": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DecisionTreeClassifier(random_state=42)),
            ]),
            {
                "model__max_depth": [5, 10, 15],
                "model__min_samples_split": [5, 10],
                "model__min_samples_leaf": [5, 10],
            },
        ),
        "Artificial Neural Network": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", MLPClassifier(
                    max_iter=500,
                    random_state=42,
                    early_stopping=False,
                    n_iter_no_change=15,
                    tol=1e-3,
                )),
            ]),
            {
                "model__hidden_layer_sizes": [(30,), (50,), (100,), (50, 25)],
                "model__alpha": [0.01, 0.1, 1.0],
            },
        ),
    }


def _features_3_sec_models():
    """Models and parameter grids specifically for the 3-second dataset.

    This dataset has many more samples than the 30-second dataset, so the
    grids are deliberately different and kept moderate to control runtime.
    """
    return {
        "Logistic Regression": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, solver="lbfgs")),
            ]),
            {"model__C": [0.03, 0.1, 0.3, 1.0, 3.0]},
        ),
        "k-NN": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier()),
            ]),
            {
                "model__n_neighbors": [5, 9, 15, 25],
                "model__weights": ["distance"],
                "model__p": [1, 2],
            },
        ),
        "Decision Tree": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DecisionTreeClassifier(random_state=42)),
            ]),
            {
                "model__max_depth": [10, 20, None],
                "model__min_samples_split": [5, 10, 20],
                "model__min_samples_leaf": [2, 5, 10],
            },
        ),
        "Artificial Neural Network": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", MLPClassifier(
                    solver="adam",
                    learning_rate_init=0.001,
                    max_iter=500,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=15,
                    tol=1e-3,
                )),
            ]),
            {
                "model__hidden_layer_sizes": [(50,), (100,), (100, 50)],
                "model__alpha": [0.001, 0.01, 0.1],
            },
        ),
    }


def get_models(dataset, n_features):
    """Return the model pipelines and parameter grids for one dataset."""
    if dataset == "GenreAll":
        return _genreall_models(n_features)
    if dataset == "features-30-sec":
        return _features_30_sec_models()
    if dataset == "features-3-sec":
        return _features_3_sec_models()
    raise ValueError(f"Unknown dataset: {dataset}")
