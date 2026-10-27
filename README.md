# Machine Learning for Sound and Music (MLSM)

Repositorio de prácticas de la asignatura **Machine Learning for Sound and Music**.

## Contenido

### Lab 1 — Clasificación de géneros musicales

La carpeta [`Lab1/`](Lab1/) contiene el notebook de ejecución, las definiciones de modelos y los documentos de apoyo.

- [`run_lab1.ipynb`](Lab1/run_lab1.ipynb): notebook para ejecutar los experimentos.
- [`run_lab1.py`](Lab1/run_lab1.py): funciones y lógica de ejecución de los experimentos.
- [`model_definitions.py`](Lab1/model_definitions.py): definiciones de los modelos.
- [`ResearchQuestions.pdf`](Lab1/ResearchQuestions.pdf): preguntas de investigación.

Consulta los archivos de la práctica para conocer los datos y dependencias necesarios.

### Lab 2 — Clasificación de géneros musicales con CNNs

La carpeta [`Lab2/`](Lab2/) contiene dos notebooks independientes:

1. [**`lab2_task1.ipynb`**](Lab2/lab2_task1.ipynb): entrenamiento y comparación de configuraciones de redes convolucionales (CNN) desde cero.
2. [**`lab2_task2.ipynb`**](Lab2/lab2_task2.ipynb): clasificación usando VGG16 preentrenada en ImageNet.

#### Datos de Lab 2

Descarga el conjunto de imágenes `images_original` indicado en el guion de la práctica y colócalo en `Lab2/Data/images_original/`, conservando una subcarpeta por género. Si utilizas otra ubicación, actualiza la variable `DATA_DIR` en el notebook correspondiente.

#### Ejecución de Lab 2

Abre cada notebook en Jupyter o Google Colab y ejecuta sus celdas en orden. Se recomienda usar una GPU, especialmente para los experimentos con VGG16. Cada notebook se ejecuta por separado; Task 2 no depende de Task 1.

## Tecnologías

Las prácticas utilizan herramientas de Python para aprendizaje automático y procesamiento de datos, incluyendo scikit-learn y TensorFlow/Keras, según la práctica.
