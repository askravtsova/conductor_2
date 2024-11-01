"""
File: tuning.py
Author: Anna Kravtsova written for Thesis (2024)
Date: September, 2024
Description: tuning of the models for big table in report
"""
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Dense, Dropout, Flatten, SimpleRNN

# Define model functions
def create_lstm_model(input_shape, dropout_rate):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=False),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    return model

def create_cnn_model(input_shape, dropout_rate):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    return model

def create_cnn_lstm_model(input_shape, dropout_rate):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(128, return_sequences=False),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    return model

def create_rnn_model(input_shape, dropout_rate):
    model = Sequential([
        SimpleRNN(128, input_shape=input_shape, return_sequences=False),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    return model

# tuning pars
def tune_model(model_fn, X, y, learning_rates=[0.001, 0.0005], batch_sizes=[16, 32], dropout_rates=[0.5]):
    results = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            for dropout_rate in dropout_rates:
                fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
                
                print(f"Tuning with Learning Rate: {lr}, Batch Size: {batch_size}, Dropout Rate: {dropout_rate}")
                
                for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Create and compile model with current hyperparameters
                    model = model_fn((X_train.shape[1], X_train.shape[2]), dropout_rate)
                    model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                    
                    # Early stopping to prevent overfitting
                    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                    
                    # Train model
                    model.fit(X_train, y_train, epochs=30, batch_size=batch_size, validation_data=(X_val, y_val),
                              callbacks=[early_stopping], verbose=0)
                    
                    # Evaluate model
                    y_val_pred = np.argmax(model.predict(X_val), axis=1)
                    accuracy = accuracy_score(y_val, y_val_pred)
                    precision = precision_score(y_val, y_val_pred, average='weighted')
                    recall = recall_score(y_val, y_val_pred, average='weighted')
                    f1 = f1_score(y_val, y_val_pred, average='weighted')
                    
                    # Store fold results
                    fold_metrics['accuracy'].append(accuracy)
                    fold_metrics['precision'].append(precision)
                    fold_metrics['recall'].append(recall)
                    fold_metrics['f1_score'].append(f1)
                
                # Calculate average performance across folds
                avg_metrics = {metric: np.mean(scores) for metric, scores in fold_metrics.items()}
                avg_metrics.update({'learning_rate': lr, 'batch_size': batch_size, 'dropout_rate': dropout_rate})
                
                # Store results
                results.append(avg_metrics)
                print(f"Avg Metrics: {avg_metrics}")
    
    # Convert results to a DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    results_df.to_csv('tuning_results.csv', index=False)
    return results_df

# Load data
# Function to load all .npy files from the processed data directories (adjusted from your existing function)
def load_all_npy_files(processed_data_dir, max_seq_len=100):
    X = []  # List to store landmark data
    y = []  # List to store corresponding labels
    
    for root, dirs, files in os.walk(processed_data_dir):
        for npy_file in files:
            if npy_file.endswith('.npy'):
                file_path = os.path.join(root, npy_file)
                landmarks = np.load(file_path)

                # Pad or truncate sequences to max_seq_len
                if len(landmarks) > max_seq_len:
                    landmarks = landmarks[:max_seq_len]
                else:
                    landmarks = np.pad(landmarks, ((0, max_seq_len - len(landmarks)), (0, 0)), 'constant')
                
                landmarks = landmarks.reshape(max_seq_len, -1)
                
                # Assign labels based on folder names or file naming convention
                label = os.path.basename(root)
                if label == "2_4_60bpm":
                    y.append(0)
                elif label == "3_4_60bpm":
                    y.append(1)
                elif label == "4_4_60bpm":
                    y.append(2)

                X.append(landmarks)

    return np.array(X), np.array(y)

# Load the data
processed_data_dir = 'data/processed_data'
X, y = load_all_npy_files(processed_data_dir, max_seq_len=100)

# Verify shapes
print(f"Data shape: {X.shape}, Labels shape: {y.shape}")

# Define model functions and names
model_functions = {
    "LSTM": create_lstm_model,
    "CNN": create_cnn_model,
    "CNN-LSTM": create_cnn_lstm_model,
    "RNN": create_rnn_model
}

# Run tuning for each model
all_results = {}
for model_name, model_fn in model_functions.items():
    print(f"\nTuning {model_name} model...")
    model_results = tune_model(model_fn, X, y)
    all_results[model_name] = model_results

# Save all results for each model in a combined file for easy comparison      
for model_name, results_df in all_results.items():
    results_df.to_csv(f'{model_name}_tuning_results.csv', index=False)

