# Lab 2 — Music Genre Classification with CNNs

This folder contains the implementation for MLSM Lab 2. **Nothing in `Lab1/` is modified by this lab work.**

## Contents
- `lab2_cnn_genre_classification.ipynb`: full notebook for the two required tasks:
  1. Compare simple CNN configurations on the provided mel-spectrogram images.
  2. Fine-tune an ImageNet-pretrained VGG16 model as a transfer-learning baseline.
- `README.md`: setup and execution notes.

## Dataset
Use the files linked in the Lab 2 handout: https://drive.google.com/drive/folders/1ijzsLkkSZHZW2JDaoH2fAm6f9IwMg8dk?usp=drive_link

Extract the `images_original` directory, preserving one subfolder per genre. The notebook expects image files under each genre folder. In the notebook, set `DATA_DIR` to the location of `images_original`. The CSV descriptor files are not used for the CNN tasks.

## Run
Open the notebook in Google Colab or Jupyter and run the cells from top to bottom. A GPU is recommended. In Colab, upload a ZIP containing `images_original/` using the optional upload cell, extract it, and set `DATA_DIR` to the extracted path. VGG16 downloads pretrained ImageNet weights on first use, so internet access is needed.

The notebook uses a single shuffled, stratified 80/20 train/test split for both model families. A validation split is taken from the training partition for model selection. It saves the final test comparison to `lab2_test_results.csv` and the CNN configuration results to `lab2_cnn_configuration_results.csv`.

**Important:** VGG16 is used here as a pretrained spectrogram-image model; it is not the audio-waveform VGGish model mentioned as an example in the handout.