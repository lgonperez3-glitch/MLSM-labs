"""Core logic for the interactive MLSM Lab 1 notebook.

The notebook imports this module and keeps only prompts and high-level calls.
Model pipelines and parameter grids are defined in model_definitions.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

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


def resolve_lab_dir():
    """Find Lab1 whether Jupyter starts in the repository root or Lab1."""
    cwd = Path.cwd()
    if (cwd / "model_definitions.py").exists() and (cwd / "data").exists():
        return cwd
    if (cwd / "Lab1" / "model_definitions.py").exists():
        return cwd / "Lab1"
    raise FileNotFoundError("No se encuentra Lab1/model_definitions.py ni su carpeta data/.")


def load_dataset(dataset_choice, data_dir=None):
    """Load a dataset and return its name, features, labels and class counts."""
    if dataset_choice not in DATASETS:
        raise ValueError("Dataset no válido: selecciona 1, 2 o 3.")
    name, filename, target = DATASETS[dataset_choice]
    csv_path = Path(data_dir or resolve_lab_dir() / "data") / filename
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encuentra el archivo: {csv_path}")

    df = pd.read_csv(csv_path, skipinitialspace=True).dropna(subset=[target])
    y = df[target].copy()
    X = df.drop(columns=[target]).copy()
    for column in X.select_dtypes(include=["object", "string"]).columns:
        values = X[column].dropna().unique()
        if len(values) == 2:
            X[column] = X[column].map({values[0]: 0, values[1]: 1})
    X = X.select_dtypes(include=np.number)
    if X.empty:
        raise ValueError(f"No se encontraron características numéricas en {filename}.")
    return name, filename, X, y, y.value_counts().rename("Muestras")


def select_models(dataset_name, n_features, choice):
    """Return the selected model pipelines and parameter grids."""
    from model_definitions import get_models

    models = get_models(dataset_name, n_features)
    choice = choice.strip().lower()
    if choice == "all":
        return models
    if choice not in MODEL_OPTIONS:
        raise ValueError("Modelo no válido: selecciona l, k, d, a o all.")
    name = MODEL_OPTIONS[choice]
    if name not in models:
        raise ValueError(f"{name} no está definido para {dataset_name}.")
    return {name: models[name]}


def evaluate_models(X, y, selected_models, cv=None):
    """Run grid search and return all parameter configurations as a DataFrame."""
    cv = cv or StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    rows = []
    for name, (pipeline, param_grid) in selected_models.items():
        search = GridSearchCV(
            pipeline, param_grid, scoring="accuracy", cv=cv,
            return_train_score=True, n_jobs=-1, error_score="raise"
        )
        search.fit(X, y)
        for i, params in enumerate(search.cv_results_["params"]):
            train_acc = search.cv_results_["mean_train_score"][i]
            cv_acc = search.cv_results_["mean_test_score"][i]
            rows.append({
                "Algorithm": name,
                "Parameters": params,
                "Train accuracy": train_acc,
                "CV accuracy": cv_acc,
                "Overfitting gap": train_acc - cv_acc,
            })
    if not rows:
        raise ValueError("No se ha evaluado ninguna configuración.")
    return pd.DataFrame(rows).sort_values(
        ["CV accuracy", "Overfitting gap"], ascending=[False, True]
    ).reset_index(drop=True)


def get_confusion_data(X, y, selected_models, results_df, row_index, cv=None):
    """Return out-of-fold confusion matrix and metadata for the selected result row."""
    if not 0 <= row_index < len(results_df):
        raise IndexError(f"La fila debe estar entre 0 y {len(results_df) - 1}.")
    cv = cv or StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    row = results_df.iloc[row_index]
    name, params = row["Algorithm"], row["Parameters"]
    estimator = clone(selected_models[name][0]).set_params(**params)
    predictions = cross_val_predict(estimator, X, y, cv=cv, n_jobs=-1)
    classes = np.unique(y)
    matrix = confusion_matrix(y, predictions, labels=classes)
    return matrix, classes, row


def create_cv():
    """Create the standard stratified 10-fold cross-validation splitter."""
    return StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


def display_results(dataframe):
    """Display a DataFrame, formatting evaluation metrics when present."""
    if {"Train accuracy", "CV accuracy", "Overfitting gap"}.issubset(dataframe.columns):
        display(dataframe.style.format({
            "Train accuracy": "{:.4f}",
            "CV accuracy": "{:.4f}",
            "Overfitting gap": "{:.4f}",
        }))
    else:
        display(dataframe)


def plot_confusion_matrix(matrix, classes, title, parameters):
    """Plot the confusion matrix with dataset/model title and parameters as a footer."""
    display = ConfusionMatrixDisplay(matrix, display_labels=classes)
    display.plot(xticks_rotation="vertical", values_format="d", colorbar=False)
    fig = display.figure_
    fig.suptitle(title, fontsize=12, y=0.98)
    fig.text(
        0.5, 0.015, f"Parameters: {parameters}",
        ha="center", va="bottom", fontsize=9, wrap=True
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    plt.show()
    return fig


def save_run_outputs(results_df, dataset_name, selected_models, fig=None):
    """Optionally export the results table, run configuration, and confusion figure."""
    answer = input("¿Quieres guardar los resultados y la figura? [y/n]: ").strip().lower()
    while answer not in {"y", "n"}:
        answer = input("Introduce y (sí) o n (no): ").strip().lower()
    if answer == "n":
        print("No se ha guardado ningún archivo. Puedes volver a ejecutar esta celda más tarde.")
        return

    if fig is None:
        print("No hay ninguna figura de matriz de confusión disponible. Genera la matriz antes de guardar.")
        return

    results_dir = resolve_lab_dir() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    while True:
        base_name = input("Nombre base de los archivos (sin extensión): ").strip()
        if not base_name or Path(base_name).name != base_name or base_name in {".", ".."}:
            print("Introduce un nombre válido, sin carpetas ni rutas.")
            continue
        excel_path = results_dir / f"{base_name}.xlsx"
        figure_path = results_dir / f"{base_name}.png"
        existing = [path.name for path in (excel_path, figure_path) if path.exists()]
        if existing:
            print(f"Ya existe: {', '.join(existing)}. Elige otro nombre.")
            continue
        break

    export_df = results_df.copy()
    export_df["Parameters"] = export_df["Parameters"].map(str)
    config_rows = []
    for model_name, (_, param_grid) in selected_models.items():
        config_rows.append({
            "Dataset": dataset_name,
            "Selected model": model_name,
            "Parameter grid": str(param_grid),
        })
    config_df = pd.DataFrame(config_rows)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Results")
        config_df.to_excel(writer, index=False, sheet_name="Configuration")
        for sheet_name, dataframe in (("Results", export_df), ("Configuration", config_df)):
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                width = min(max(max(len(str(cell.value or "")) for cell in column_cells) + 2, 12), 60)
                worksheet.column_dimensions[column_cells[0].column_letter].width = width
            if sheet_name == "Results":
                for column in ("Train accuracy", "CV accuracy", "Overfitting gap"):
                    if column in dataframe.columns:
                        col_idx = dataframe.columns.get_loc(column) + 1
                        for cells in worksheet.iter_cols(min_col=col_idx, max_col=col_idx, min_row=2):
                            for cell in cells:
                                cell.number_format = "0.0000"

    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    print(f"Excel guardado en: {excel_path}")
    print(f"Figura guardada en: {figure_path}")
