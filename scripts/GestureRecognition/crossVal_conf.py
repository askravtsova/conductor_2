"""
File: crossval plots.py
Author: Anna Kravtsova written for Thesis (2024)
Date: September, 2024
Description: pcross validation plots
"""
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import clone_model, load_model

# Load the trained models
cnn_lstm_model = load_model('saved_models/CNN-LSTM_model.h5')
cnn_model = load_model('saved_models/CNN_model.h5')

# Function to load and preprocess dataset
def load_all_npy_files(directory, max_seq_len=100, target_feature_size=63):
    X = []
    y = []
    
    for root, dirs, files in os.walk(directory):
        for npy_file in files:
            if npy_file.endswith('.npy'):
                file_path = os.path.join(root, npy_file)
                landmarks = np.load(file_path)

                if landmarks.shape[1] * landmarks.shape[2] != target_feature_size:
                    landmarks = landmarks[:, :target_feature_size // landmarks.shape[2], :landmarks.shape[2]]
                landmarks = landmarks.reshape(-1, target_feature_size)
                
                if len(landmarks) > max_seq_len:
                    landmarks = landmarks[:max_seq_len]
                else:
                    landmarks = np.pad(landmarks, ((0, max_seq_len - len(landmarks)), (0, 0)), 'constant')
                
                X.append(landmarks)
                
                label = os.path.basename(root)
                if label in ["2_4_60bpm", "2_4_90bpm"]:
                    y.append(0)
                elif label in ["3_4_60bpm", "3_4_90bpm"]:
                    y.append(1)
                elif label in ["4_4_60bpm", "4_4_90bpm"]:
                    y.append(2)

    return np.array(X), np.array(y)

# Paths to datasets
original_dataset_path = 'data/processed_data'
bpm_90_dataset_path = 'data/bigger_dataset/processed_data'

# Load datasets
X_original, y_original = load_all_npy_files(original_dataset_path)
X_90bpm, y_90bpm = load_all_npy_files(bpm_90_dataset_path)

# K-Fold  Cross Validation and Confusion Matrix Evaluation
def kfold_evaluate_model(model, X, y, n_splits=5, dataset_name=""):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1_score": []}
    confusion_matrices = []

    for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
        print(f"{dataset_name} - Fold {fold}:")

        # Split data into training and validation for this fold
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # clone for each fold
        model_fold = clone_model(model)
        model_fold.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train model on training data
        model_fold.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

        # Predictions and confusion matrix
        y_val_pred = np.argmax(model_fold.predict(X_val), axis=1)
        cm = confusion_matrix(y_val, y_val_pred)
        confusion_matrices.append(cm)

        #  metrics
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred, average='weighted')
        recall = recall_score(y_val, y_val_pred, average='weighted')
        f1 = f1_score(y_val, y_val_pred, average='weighted')

        # Append metrics
        fold_metrics["accuracy"].append(accuracy)
        fold_metrics["precision"].append(precision)
        fold_metrics["recall"].append(recall)
        fold_metrics["f1_score"].append(f1)

        print(f"Fold {fold} Metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1:.4f}")

    # Average metrics across folds
    avg_metrics = {metric: np.mean(scores) for metric, scores in fold_metrics.items()}
    std_metrics = {metric: np.std(scores) for metric, scores in fold_metrics.items()}
    print(f"\n{dataset_name} - Average Metrics Across Folds:")
    print(pd.DataFrame([avg_metrics, std_metrics], index=["Mean", "Std"]))

    # Sum confusion matrices to get an aggregate matrix
    aggregate_cm = sum(confusion_matrices)

    # Display aggregate confusion matrix
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix=aggregate_cm, display_labels=["2/4", "3/4", "4/4"]).plot(cmap=plt.cm.Blues)
    plt.title(f"{dataset_name} - Aggregate Confusion Matrix Across Folds")
    plt.show()

    return avg_metrics, std_metrics, confusion_matrices

# Run K-Foldn Cross Validation omn both models and both datasets
print("\nEvaluating CNN-LSTM Model on Original Dataset:")
cnn_lstm_metrics_original = kfold_evaluate_model(cnn_lstm_model, X_original, y_original, n_splits=5, dataset_name="CNN-LSTM Original")

print("\nEvaluating CNN-LSTM Model on 90 BPM Dataset:")
cnn_lstm_metrics_90bpm = kfold_evaluate_model(cnn_lstm_model, X_90bpm, y_90bpm, n_splits=5, dataset_name="CNN-LSTM 90 BPM")

print("\nEvaluating CNN Model on Original Dataset:")
cnn_metrics_original = kfold_evaluate_model(cnn_model, X_original, y_original, n_splits=5, dataset_name="CNN Original")

print("\nEvaluating CNN Model on 90 BPM Dataset:")
cnn_metrics_90bpm = kfold_evaluate_model(cnn_model, X_90bpm, y_90bpm, n_splits=5, dataset_name="CNN 90 BPM")
