"""
File: AM metrics epochst.py
Author: Anna Kravtsova written for Thesis (2024)
Date: September, 2024
Description:over epoch trainibgf for report data
"""
import numpy as np
import os
import json
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, SimpleRNN

# Load data 
def load_all_npy_files(processed_data_dir, max_seq_len=100):
    X, y = [], []  # Lists to store data and labels
    for root, dirs, files in os.walk(processed_data_dir):
        for npy_file in files:
            if npy_file.endswith('.npy'):
                file_path = os.path.join(root, npy_file)
                landmarks = np.load(file_path)
                if len(landmarks) > max_seq_len:
                    landmarks = landmarks[:max_seq_len]
                else:
                    landmarks = np.pad(landmarks, ((0, max_seq_len - len(landmarks)), (0, 0)), 'constant')
                landmarks = landmarks.reshape(max_seq_len, -1)
                label = os.path.basename(root)
                if label == "2_4_60bpm":
                    y.append(0)
                elif label == "3_4_60bpm":
                    y.append(1)
                elif label == "4_4_60bpm":
                    y.append(2)
                X.append(landmarks)
    return np.array(X), np.array(y)

# Define model architectures
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    return model

def create_cnn_model(input_shape):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    return model

def create_cnn_lstm_model(input_shape):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(128, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    return model

def create_rnn_model(input_shape):
    model = Sequential([
        SimpleRNN(128, input_shape=input_shape, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    return model

# experiment traning over epochs
def train_and_evaluate_model(model_fn, X_train, y_train, X_val, y_val):
    model = model_fn((X_train.shape[1], X_train.shape[2]))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=60, batch_size=16)
    
    #  accuracy and loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    return model, history

# K- Cross-Validation
def k_fold_validation(model_fn, X, y, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"Starting fold {fold + 1}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = model_fn((X_train.shape[1], X_train.shape[2]))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)
        y_val_pred = np.argmax(model.predict(X_val), axis=1)
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred, average='weighted')
        recall = recall_score(y_val, y_val_pred, average='weighted')
        f1 = f1_score(y_val, y_val_pred, average='weighted')
        fold_metrics['accuracy'].append(accuracy)
        fold_metrics['precision'].append(precision)
        fold_metrics['recall'].append(recall)
        fold_metrics['f1_score'].append(f1)
        print(f"Fold {fold + 1} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    avg_metrics = {k: np.mean(v) for k, v in fold_metrics.items()}
    print("\nCross-Validation Results (Averaged):")
    for metric, avg in avg_metrics.items():
        print(f"{metric.capitalize()}: {avg:.4f}")
    return avg_metrics

#  data
processed_data_dir = 'data/processed_data'
X, y = load_all_npy_files(processed_data_dir, max_seq_len=100)

# Split data for training over epochs
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#  model functions and names
models = {
    "LSTM": create_lstm_model,
    "CNN": create_cnn_model,
    "CNN-LSTM": create_cnn_lstm_model,
    "RNN": create_rnn_model
}

#  experiments for each model
results = {}
for model_name, model_fn in models.items():
    print(f"\nTraining {model_name} model over epochs...")
    trained_model, history = train_and_evaluate_model(model_fn, X_train, y_train, X_val, y_val)

    print(f"Performing K-Fold cross-validation on {model_name} model...")
    kfold_results = k_fold_validation(model_fn, X, y)
    results[model_name] = kfold_results


# Display all results
print("\nFinal Model Results Summary:")
for model_name, metrics in results.items():
    print(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1 Score: {metrics['f1_score']:.4f}")
