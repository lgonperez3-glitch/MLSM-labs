# Lab 2 — Music Genre Classification with CNNs

Lab 2 está organizado en dos notebooks independientes:

- **Task 1 — [`lab2_task1.ipynb`](lab2_task1.ipynb):** entrenamiento desde cero y comparación de 20 configuraciones de CNN. Incluye selección del modelo mediante validación y evaluación final en el conjunto de test.
- **Task 2 — [`lab2_task2.ipynb`](lab2_task2.ipynb):** clasificación de géneros musicales usando VGG16 preentrenada en ImageNet.

## Datos

Descarga `images_original` desde el enlace de datos del guion y colócala en `Lab2/Data/images_original/`, conservando una subcarpeta por género. Si la carpeta está en otra ubicación, cambia `DATA_DIR` en el notebook que vayas a ejecutar.

## Ejecución

Abre el notebook correspondiente en Jupyter o Google Colab y ejecuta las celdas en orden. Ejecuta Task 1 y Task 2 por separado; no dependen entre sí. Se recomienda GPU, especialmente para Task 2 con VGG16.
