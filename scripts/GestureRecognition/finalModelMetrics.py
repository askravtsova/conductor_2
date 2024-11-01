"""
File: final modelMetrics.py
Author: Anna Kravtsova written for Thesis (2024)
Date: September, 2024
Description: final model metrics  for big table in report
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model

# Load the trained models
cnn_lstm_model = load_model('saved_models/CNN-LSTM_model.h5')
cnn_model = load_model('saved_models/CNN_model.h5')

def load_all_npy_files(directory, max_seq_len=100, target_feature_size=63):
    X = []
    y = []
    
    # Load the files as per your original function
    for root, dirs, files in os.walk(directory):
        for npy_file in files:
            if npy_file.endswith('.npy'):
                file_path = os.path.join(root, npy_file)
                landmarks = np.load(file_path)
                
                # Print shape for debugging
                print(f"Original shape of landmarks: {landmarks.shape}")
                
                # Check if landmarks is 3D or 2D
                if landmarks.ndim == 3:
                    # Ensure each frame has the target number of features (target_feature_size)
                    if landmarks.shape[1] * landmarks.shape[2] != target_feature_size:
                        landmarks = landmarks[:, :target_feature_size // landmarks.shape[2], :landmarks.shape[2]]
                    landmarks = landmarks.reshape(-1, target_feature_size)
                
                elif landmarks.ndim == 2:
                    # Ensure each frame has the target number of features (target_feature_size)
                    if landmarks.shape[1] < target_feature_size:
                        landmarks = np.pad(landmarks, ((0, 0), (0, target_feature_size - landmarks.shape[1])), 'constant')
                    elif landmarks.shape[1] > target_feature_size:
                        landmarks = landmarks[:, :target_feature_size]

                # Pad or truncate sequences to max_seq_len
                if landmarks.shape[0] > max_seq_len:
                    landmarks = landmarks[:max_seq_len]
                else:
                    landmarks = np.pad(landmarks, ((0, max_seq_len - landmarks.shape[0]), (0, 0)), 'constant')
                
                print(f"Processed shape of landmarks: {landmarks.shape}")
                
                X.append(landmarks)
                
                # Map folder names to labels
                label = os.path.basename(root)
                if label in ["2_4_60bpm", "2_4_90bpm"]:
                    y.append(0)
                elif label in ["3_4_60bpm", "3_4_90bpm"]:
                    y.append(1)
                elif label in ["4_4_60bpm", "4_4_90bpm"]:
                    y.append(2)

    return np.array(X), np.array(y)


# Paths to datasets
dataset_a_path = 'data/processed_data'
dataset_b_path = 'data/bigger_dataset/processed_data'

# Load both datasets
X_a, y_a = load_all_npy_files(dataset_a_path)
X_b, y_b = load_all_npy_files(dataset_b_path)

# Function to evaluate a model on a given dataset
def evaluate_model(model, X, y):
    y_pred = np.argmax(model.predict(X), axis=1)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Evaluate CNN-LSTM on both datasets
cnn_lstm_a_metrics = evaluate_model(cnn_lstm_model, X_a, y_a)
cnn_lstm_b_metrics = evaluate_model(cnn_lstm_model, X_b, y_b)

# Evaluate CNN on both datasets
cnn_a_metrics = evaluate_model(cnn_model, X_a, y_a)
cnn_b_metrics = evaluate_model(cnn_model, X_b, y_b)

# Organize results into a DataFrame
evaluation_results = pd.DataFrame({
    "Model": ["CNN-LSTM", "CNN-LSTM", "CNN", "CNN"],
    "Dataset": ["Original", "90 BPM", "Original", "90 BPM"],
    "Accuracy": [cnn_lstm_a_metrics[0], cnn_lstm_b_metrics[0], cnn_a_metrics[0], cnn_b_metrics[0]],
    "Precision": [cnn_lstm_a_metrics[1], cnn_lstm_b_metrics[1], cnn_a_metrics[1], cnn_b_metrics[1]],
    "Recall": [cnn_lstm_a_metrics[2], cnn_lstm_b_metrics[2], cnn_a_metrics[2], cnn_b_metrics[2]],
    "F1 Score": [cnn_lstm_a_metrics[3], cnn_lstm_b_metrics[3], cnn_a_metrics[3], cnn_b_metrics[3]]
})

# Display evaluation results
print("Evaluation Results:\n", evaluation_results)

# Visualization of results
plt.figure(figsize=(12, 6))

# Plot Accuracy
plt.subplot(1, 2, 1)
for model in ["CNN-LSTM", "CNN"]:
    subset = evaluation_results[evaluation_results["Model"] == model]
    plt.plot(subset["Dataset"], subset["Accuracy"], marker='o', label=f"{model} Accuracy")
plt.title("Model Accuracy Across Datasets")
plt.xlabel("Dataset")
plt.ylabel("Accuracy")
plt.legend()

# Plot F1 Score
plt.subplot(1, 2, 2)
for model in ["CNN-LSTM", "CNN"]:
    subset = evaluation_results[evaluation_results["Model"] == model]
    plt.plot(subset["Dataset"], subset["F1 Score"], marker='o', label=f"{model} F1 Score")
plt.title("Model F1 Score Across Datasets")
plt.xlabel("Dataset")
plt.ylabel("F1 Score")
plt.legend()

plt.tight_layout()
plt.show()
