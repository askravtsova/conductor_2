"""
File: cnn lstm compare.py
Author: Anna Kravtsova written for Thesis (2024)
Date: September, 2024
Description: comapre all metrics and cnn lstm stuff, pre loaded models pre trained
"""
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

# Load the trained models
cnn_lstm_model = load_model('saved_models/CNN-LSTM_model.h5')
cnn_model = load_model('saved_models/CNN_model.h5')

# Function to load and preprocess dataset
def load_all_npy_files(directory, max_seq_len=100, target_feature_size=63):
    X = []
    y = []
    
    # Load files in the directory
    for root, dirs, files in os.walk(directory):
        for npy_file in files:
            if npy_file.endswith('.npy'):
                file_path = os.path.join(root, npy_file)
                landmarks = np.load(file_path)

                # Adjust features to match target feature size
                if landmarks.shape[1] * landmarks.shape[2] != target_feature_size:
                    landmarks = landmarks[:, :target_feature_size // landmarks.shape[2], :landmarks.shape[2]]
                landmarks = landmarks.reshape(-1, target_feature_size)
                
                # Pad or truncate to match max_seq_len
                if len(landmarks) > max_seq_len:
                    landmarks = landmarks[:max_seq_len]
                else:
                    landmarks = np.pad(landmarks, ((0, max_seq_len - len(landmarks)), (0, 0)), 'constant')
                
                # Add data and labels
                X.append(landmarks)
                
                # Assign labels based on folder names
                label = os.path.basename(root)
                if label in ["2_4_60bpm", "2_4_90bpm"]:
                    y.append(0)
                elif label in ["3_4_60bpm", "3_4_90bpm"]:
                    y.append(1)
                elif label in ["4_4_60bpm", "4_4_90bpm"]:
                    y.append(2)

    return np.array(X), np.array(y)

#  to datasets
original_dataset_path = 'data/processed_data'
bpm_90_dataset_path = 'data/bigger_dataset/processed_data'

#  datasets
X_original, y_original = load_all_npy_files(original_dataset_path)
X_90bpm, y_90bpm = load_all_npy_files(bpm_90_dataset_path)

#  function with confusion matrix display
def evaluate_model(model, X, y, dataset_name, model_name):
    y_pred = np.argmax(model.predict(X), axis=1)
    
    #  metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    
    # Print metrics
    print(f"\n{model_name} Performance on {dataset_name} Dataset:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["2/4", "3/4", "4/4"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_name} Confusion Matrix on {dataset_name}")
    plt.show()
    
    return accuracy, precision, recall, f1

# khbi CNN-LSTM on both datasets
cnn_lstm_original_metrics = evaluate_model(cnn_lstm_model, X_original, y_original, "Original", "CNN-LSTM")
cnn_lstm_90bpm_metrics = evaluate_model(cnn_lstm_model, X_90bpm, y_90bpm, "90 BPM", "CNN-LSTM")

# evalwpb CNN on both datasets
cnn_original_metrics = evaluate_model(cnn_model, X_original, y_original, "Original", "CNN")
cnn_90bpm_metrics = evaluate_model(cnn_model, X_90bpm, y_90bpm, "90 BPM", "CNN")


results_df = pd.DataFrame({
    "Model": ["CNN-LSTM", "CNN-LSTM", "CNN", "CNN"],
    "Dataset": ["Original", "90 BPM", "Original", "90 BPM"],
    "Accuracy": [cnn_lstm_original_metrics[0], cnn_lstm_90bpm_metrics[0], cnn_original_metrics[0], cnn_90bpm_metrics[0]],
    "Precision": [cnn_lstm_original_metrics[1], cnn_lstm_90bpm_metrics[1], cnn_original_metrics[1], cnn_90bpm_metrics[1]],
    "Recall": [cnn_lstm_original_metrics[2], cnn_lstm_90bpm_metrics[2], cnn_original_metrics[2], cnn_90bpm_metrics[2]],
    "F1 Score": [cnn_lstm_original_metrics[3], cnn_lstm_90bpm_metrics[3], cnn_original_metrics[3], cnn_90bpm_metrics[3]]
})

#dispal
print("\nFinal Model Evaluation Results:")
print(results_df)
