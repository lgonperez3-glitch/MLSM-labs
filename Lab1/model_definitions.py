"""Shared model definitions for the MLSM Lab 1 datasets.

All datasets use the same pipelines and hyperparameter grids. Feature-selection
values are filtered according to the number of input features available.
"""

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


SUPPORTED_DATASETS = {"GenreAll", "features-30-sec", "features-3-sec"}

# Shared options, combined from the grids previously used for all datasets.
FEATURE_SELECTION_OPTIONS = [10, 20, 30, 40, 50, 80, 100, 130, 150, 180, 200]

PARAMETER_GRIDS = {
    "Logistic Regression": {
        "model__C": [0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 1.0, 3.0],
    },
    "k-NN": {
        "model__n_neighbors": [3, 5, 7, 9],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    },
    "Decision Tree": {
        "model__max_depth": [5, 10, 15, 20, None],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [2, 5, 10],
    },
    "Artificial Neural Network": {
        "model__hidden_layer_sizes": [
            (50,), (100,), (50, 25), (100, 50)
        ],
        "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
        "model__learning_rate_init": [0.0005, 0.001],
    },
}


def get_models(dataset, n_features):
    """Return shared pipelines and parameter grids for any supported dataset."""
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset}. Expected one of {sorted(SUPPORTED_DATASETS)}"
        )

    # Keep k below the raw feature count; "all" always remains available.
    selector_options = [
        k for k in FEATURE_SELECTION_OPTIONS if k < n_features
    ]
    selector_options.append("all")

    common_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.0)),
    ]

    models = {
        "Logistic Regression": (
            Pipeline(common_steps + [
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(score_func=f_classif)),
                ("model", LogisticRegression(max_iter=5000, solver="lbfgs")),
            ]),
            {
                "selector__k": selector_options,
                **PARAMETER_GRIDS["Logistic Regression"],
            },
        ),
        "k-NN": (
            Pipeline(common_steps + [
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(score_func=f_classif)),
                ("model", KNeighborsClassifier()),
            ]),
            {
                "selector__k": selector_options,
                **PARAMETER_GRIDS["k-NN"],
            },
        ),
        "Decision Tree": (
            Pipeline(common_steps + [
                ("selector", SelectKBest(score_func=f_classif)),
                ("model", DecisionTreeClassifier(random_state=42)),
            ]),
            {
                "selector__k": selector_options,
                **PARAMETER_GRIDS["Decision Tree"],
            },
        ),
        "Artificial Neural Network": (
            Pipeline(common_steps + [
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(score_func=f_classif)),
                ("model", MLPClassifier(
                    solver="adam",
                    max_iter=1000,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=15,
                    tol=1e-3,
                )),
            ]),
            {
                "selector__k": selector_options,
                **PARAMETER_GRIDS["Artificial Neural Network"],
            },
        ),
    }

    return models
