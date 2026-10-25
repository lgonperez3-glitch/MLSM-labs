"""Interactive runner for MLSM Lab 1.

Run from any working directory:
    python Lab1/run_lab1.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from model_definitions import get_models


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DATASETS = {
    "1": ("GenreAll", "GenreAll.csv", "genre"),
    "2": ("features-30-sec", "features_30_sec.csv", "label"),
    "3": ("features-3-sec", "features_3_sec.csv", "label"),
}

MODEL_OPTIONS = {
    "l": "Logistic Regression",
    "k": "k-NN",
    "d": "Decision Tree",
    "a": "Artificial Neural Network",
}


def choose_dataset():
    """Ask which dataset should be used."""
    print("\nSelect the dataset:")
    print("  1 -> GenreAll")
    print("  2 -> features-30-sec")
    print("  3 -> features-3-sec")

    while True:
        choice = input("Dataset [1/2/3]: ").strip().lower()
        if choice in DATASETS:
            return DATASETS[choice]
        print("Invalid option. Enter 1, 2 or 3.")


def choose_models(models):
    """Ask which model(s) should be trained."""
    print("\nSelect the model(s) to train:")
    print("  l   -> Logistic Regression")
    print("  k   -> k-NN")
    print("  d   -> Decision Tree")
    print("  a   -> Artificial Neural Network")
    print("  all -> Train all models")

    while True:
        choice = input("Model [l/k/d/a/all]: ").strip().lower()

        if choice == "all":
            return models.copy()

        if choice in MODEL_OPTIONS:
            model_name = MODEL_OPTIONS[choice]
            if model_name in models:
                return {model_name: models[model_name]}
            print(f"{model_name} is not defined for this dataset.")
        else:
            print("Invalid option. Enter l, k, d, a or all.")


def load_dataset(csv_filename, target_column):
    """Load and prepare a dataset in the same way as the Lab 1 notebooks."""
    csv_path = DATA_DIR / csv_filename
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Check that the CSV files are inside Lab1/data/."
        )

    df = pd.read_csv(csv_path, skipinitialspace=True)
    df = df.dropna(subset=[target_column])

    y = df[target_column].copy()
    X = df.drop(columns=[target_column]).copy()

    # Convert binary text columns to numeric 0/1 values.
    text_columns = X.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        values = X[column].dropna().unique()
        if len(values) == 2:
            X[column] = X[column].map({values[0]: 0, values[1]: 1})

    # Keep numeric features only (e.g. non-numeric filename columns are removed).
    X = X.select_dtypes(include=np.number)

    if X.empty:
        raise ValueError(f"No numeric features found in {csv_filename}.")

    print(f"\nDataset: {csv_filename}")
    print(f"Number of samples: {len(df)}")
    print(f"Number of numeric features: {X.shape[1]}")
    print("\nSamples per class:")
    print(y.value_counts())

    return X, y


def train_models(models, X, y, cv):
    """Run GridSearchCV and collect every evaluated configuration."""
    results = []

    for name, (pipeline, param_grid) in models.items():
        print(f"\nTraining: {name}")

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="accuracy",
            cv=cv,
            return_train_score=True,
            n_jobs=-1,
            error_score="raise",
        )
        search.fit(X, y)

        for i, params in enumerate(search.cv_results_["params"]):
            train_accuracy = search.cv_results_["mean_train_score"][i]
            cv_accuracy = search.cv_results_["mean_test_score"][i]
            results.append({
                "Algorithm": name,
                "Parameters": params,
                "Train accuracy": train_accuracy,
                "CV accuracy": cv_accuracy,
                "Overfitting gap": train_accuracy - cv_accuracy,
            })

        print(f"Best parameters: {search.best_params_}")
        print(f"Best CV accuracy: {search.best_score_:.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        ["CV accuracy", "Overfitting gap"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return results_df


def show_results(results_df):
    """Print a numbered, sorted table of evaluated configurations."""
    print("\n\nRESULTS (sorted by CV accuracy, then overfitting gap)")
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.max_colwidth", 100,
        "display.width", 220,
    ):
        print(results_df.to_string(index=True, float_format=lambda value: f"{value:.4f}"))

    print(f"\nTotal configurations evaluated: {len(results_df)}")


def choose_confusion_matrix_row(results_df):
    """Ask which result row should be used for the confusion matrix."""
    print("\nWhich result would you like to visualize as a confusion matrix?")
    print("Enter the row index shown in the results table, or q to skip.")

    while True:
        choice = input(f"Result row [0-{len(results_df) - 1}]: ").strip().lower()
        if choice == "q":
            return None
        try:
            row = int(choice)
        except ValueError:
            print("Enter a valid row number or q.")
            continue

        if 0 <= row < len(results_df):
            return row
        print(f"Row must be between 0 and {len(results_df) - 1}.")


def plot_confusion_matrix(row_index, results_df, models, X, y, cv):
    """Plot out-of-fold predictions for the selected result configuration."""
    selected_row = results_df.iloc[row_index]
    model_name = selected_row["Algorithm"]
    params = selected_row["Parameters"]

    print(f"\nSelected algorithm: {model_name}")
    print(f"Parameters: {params}")
    print(f"CV accuracy: {selected_row['CV accuracy']:.4f}")
    print(f"Overfitting gap: {selected_row['Overfitting gap']:.4f}")
    print("\nCalculating out-of-fold predictions for the confusion matrix...")

    pipeline, _ = models[model_name]
    selected_model = clone(pipeline)
    selected_model.set_params(**params)

    y_pred = cross_val_predict(
        selected_model,
        X,
        y,
        cv=cv,
        n_jobs=-1,
    )

    classes = np.unique(y)
    matrix = confusion_matrix(y, y_pred, labels=classes)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=classes,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    display.plot(
        ax=ax,
        xticks_rotation="vertical",
        values_format="d",
        colorbar=False,
    )
    ax.set_title(f"Confusion matrix - {model_name}\nResult row {row_index}")
    plt.tight_layout()
    plt.show()


def main():
    dataset_name, csv_filename, target_column = choose_dataset()
    X, y = load_dataset(csv_filename, target_column)

    # The model definitions select the appropriate pipelines and parameter grids.
    available_models = get_models(dataset_name, X.shape[1])
    selected_models = choose_models(available_models)

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results_df = train_models(selected_models, X, y, cv)

    show_results(results_df)
    row_index = choose_confusion_matrix_row(results_df)

    if row_index is not None:
        plot_confusion_matrix(row_index, results_df, selected_models, X, y, cv)
    else:
        print("Confusion matrix skipped.")


if __name__ == "__main__":
    main()
