"""
File: e1kFOLD multip.py
Author: Anna Kravtsova written for Thesis (2024)
Date: September, 2024
Description: similar to prev but more so focused on multiple pars for k fold stuff
"""
import numpy as np
import os
import json
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

# Function to load all .npy files from the processed data directories
def load_all_npy_files(processed_data_dir, max_seq_len=100):
    X = []  # List to store landmark data
    y = []  # List to store corresponding labels
    
    #  t
    for root, dirs, files in os.walk(processed_data_dir):
        for npy_file in files:
            if npy_file.endswith('.npy'):
                file_path = os.path.join(root, npy_file)
                landmarks = np.load(file_path)

                # Pad or truncate sequences to max_seq_len
                if len(landmarks) > max_seq_len:
                    landmarks = landmarks[:max_seq_len]  # Truncate longer sequences
                else:
                    landmarks = np.pad(landmarks, ((0, max_seq_len - len(landmarks)), (0, 0)), 'constant')  # Pad shorter sequences

                # Reshape landmarks to (max_seq_len, num_landmarks * 3)
                landmarks = landmarks.reshape(max_seq_len, -1)

                # Labeling based on folder names
                label = os.path.basename(root)
                if label == "2_4_60bpm":
                    y.append(0)  # Label for 2/4 time signature
                elif label == "3_4_60bpm":
                    y.append(1)  # Label for 3/4 time signature
                elif label == "4_4_60bpm":
                    y.append(2)  # Label for 4/4 time signature

                X.append(landmarks)

    return np.array(X), np.array(y)

# Define the paths for your processed data
processed_data_dir = 'data/processed_data'

# Load all the data
X, y = load_all_npy_files(processed_data_dir, max_seq_len=100)

# Define model architectures
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    return model

def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    return model

def create_cnn_lstm_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # Assuming 3 classes
    return model

def create_rnn_model(input_shape):
    model = Sequential()
    model.add(tf.keras.layers.SimpleRNN(128, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # Assuming 3 classes
    return model



# Training and evaluation function with k-fold cross-validation
def evaluate_model_kfold(model_fn, X, y, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\nStarting fold {fold + 1}/{n_splits}...")
        
        # Split data for current fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Initialize model
        model = model_fn(input_shape=(X_train.shape[1], X_train.shape[2]))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train model
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=16, callbacks=[early_stopping], verbose=0)

        # Evaluate model on validation set
        y_val_pred = np.argmax(model.predict(X_val), axis=1)
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred, average='weighted')
        recall = recall_score(y_val, y_val_pred, average='weighted')
        f1 = f1_score(y_val, y_val_pred, average='weighted')

        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
        
        print(f"Fold {fold + 1} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Compute average metrics across all folds
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    std_metrics = {k: np.std(v) for k, v in metrics.items()}
    
    print("\nCross-Validation Results:")
    for metric, avg in avg_metrics.items():
        print(f"{metric.capitalize()}: {avg:.4f} ± {std_metrics[metric]:.4f}")

    return avg_metrics, std_metrics

# Main experiment function
def run_experiments(X, y):
    models = {
        "LSTM": create_lstm_model,
        "CNN": create_cnn_model,
        "CNN-LSTM": create_cnn_lstm_model,
        "RNN": create_rnn_model
    }
    results = {}

    for model_name, model_fn in models.items():
        print(f"\nRunning k-fold cross-validation for {model_name} model...")
        avg_metrics, std_metrics = evaluate_model_kfold(model_fn, X, y, n_splits=5)
        results[model_name] = {"avg_metrics": avg_metrics, "std_metrics": std_metrics}


# Load data (as done previously)
processed_data_dir = 'data/processed_data'
X, y = load_all_npy_files(processed_data_dir, max_seq_len=100)

# Run experiments with multiple models poetnt8ially othe dara set LATER
run_experiments(X, y)
