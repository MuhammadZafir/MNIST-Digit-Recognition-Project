# MNIST-Digit-Recognition-Project

This repository contains the code for an MNIST Digit Recognition project, implementing machine learning models using Scikit-learn and deploying a Gradio web application for interactive digit classification.

## Project Structure

- mnist_classification.ipynb: A Jupyter Notebook containing the full data exploration, preprocessing, model training (SGD Classifier, Random Forest, KNeighborsClassifier), evaluation, and error visualization. This notebook is designed for step-by-step execution and analysis.
- app.py: The Python script for the Gradio web application, allowing users to draw digits and get real-time predictions.
  requirements.txt: A list of Python dependencies required to run the project.

Features

-Data Loading & Preprocessing: Loads the MNIST dataset, normalizes pixel values, and applies `StandardScaler` for optimal model performance.
-Model Training: Trains `SGDClassifier` (for both binary and multiclass tasks) and `RandomForestClassifier`. Includes an exploration of `KNeighborsClassifier` with hyperparameter tuning and data augmentation.
-Evaluation:   Provides accuracy, confusion matrices, classification reports, precision, recall, F1-score, and ROC curves (for binary classification).
-Error Analysis:   Visualizes worst-case misclassifications to identify common error patterns.
-Data Augmentation:   Implements image shifting to expand the training set and improve model robustness.
-Gradio Web App:   An interactive web interface for drawing digits and getting real-time predictions from the trained model.

