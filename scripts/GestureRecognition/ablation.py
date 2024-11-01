"""
File: abltaion .py
Author: Anna Kravtsova written for Thesis (2024)
Date: September, 2024
Description: ablation section part big kahuna
"""
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.models import load_model

# og CNN-LSTM Model adjust drop OUT
def create_cnn_lstm_model(input_shape, dropout_rate=0.5):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(128, return_sequences=False),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')  # Assuming 3 gesture classes
    ])
    return model

# CNN Model (No LSTM Layer)
def create_cnn_model(input_shape, dropout_rate=0.5):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    return model

# CNN-LSTM Model with No Dropout
def create_cnn_lstm_no_dropout(input_shape, **kwargs):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(128, return_sequences=False),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    return model

# eval functuon of all the stuff
def evaluate_ablation_model(model_fn, X, y, n_splits=5, model_name="", dropout_rate=0.5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1_score": []}
    
    for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
        print(f"{model_name} (Dropout={dropout_rate}) - Fold {fold}:")

        # Split data into training and validation for this fold
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Create and compile the model
        if dropout_rate is not None:
            model = model_fn((X_train.shape[1], X_train.shape[2]), dropout_rate=dropout_rate)
        else:
            model = model_fn((X_train.shape[1], X_train.shape[2]))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # train model on training data
        model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

        # Predictions on validation set
        y_val_pred = np.argmax(model.predict(X_val), axis=1)

        # alculate metrics
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred, average='weighted')
        recall = recall_score(y_val, y_val_pred, average='weighted')
        f1 = f1_score(y_val, y_val_pred, average='weighted')

        # sTore metrics
        fold_metrics["accuracy"].append(accuracy)
        fold_metrics["precision"].append(precision)
        fold_metrics["recall"].append(recall)
        fold_metrics["f1_score"].append(f1)

    # avg metrics /folds
    avg_metrics = {metric: np.mean(scores) for metric, scores in fold_metrics.items()}
    print(f"\n{model_name} (Dropout={dropout_rate}) - Average Metrics Across Folds:")
    print(pd.DataFrame([avg_metrics], index=[f"{model_name} (Dropout={dropout_rate})"]))
    
    # saving and prihjt
    return {
        "Model": model_name,
        "Dataset": "Original" if 'Original' in model_name else "90 BPM",
        "Dropout Rate": dropout_rate,
        "Accuracy": avg_metrics["accuracy"],
        "Precision": avg_metrics["precision"],
        "Recall": avg_metrics["recall"],
        "F1 Score": avg_metrics["f1_score"]
    }
    
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


# Load datasets
original_dataset_path = 'data/processed_data'
bpm_90_dataset_path = 'data/bigger_dataset/processed_data'
X, y = load_all_npy_files(original_dataset_path)
X_90, y_90 = load_all_npy_files(bpm_90_dataset_path)

# Store all results in a list
all_results = []

# Run evaluations for each ablated model and store results

# Original CNN-LSTM Model with default dropout rate (0.5)
all_results.append(evaluate_ablation_model(create_cnn_lstm_model, X, y, model_name="Original CNN-LSTM", dropout_rate=0.5))
all_results.append(evaluate_ablation_model(create_cnn_lstm_model, X_90, y_90, model_name="Original CNN-LSTM (90BPM)", dropout_rate=0.5))

# CNN Model (No LSTM)
all_results.append(evaluate_ablation_model(create_cnn_model, X, y, model_name="CNN Only", dropout_rate=0.5))
all_results.append(evaluate_ablation_model(create_cnn_model, X_90, y_90, model_name="CNN Only (90BPM)", dropout_rate=0.5))

# CNN-LSTM Model with No Dropout
all_results.append(evaluate_ablation_model(create_cnn_lstm_no_dropout, X, y, model_name="CNN-LSTM No Dropout"))
all_results.append(evaluate_ablation_model(create_cnn_lstm_no_dropout, X_90, y_90, model_name="CNN-LSTM No Dropout (90BPM)"))

# Additional: Evaluate CNN-LSTM Model with Different Dropout Rates
dropout_rates = [0.3, 0.4, 0.6, 0.7]
for rate in dropout_rates:
    all_results.append(evaluate_ablation_model(create_cnn_lstm_model, X, y, model_name="CNN-LSTM", dropout_rate=rate))
    all_results.append(evaluate_ablation_model(create_cnn_lstm_model, X_90, y_90, model_name="CNN-LSTM (90BPM)", dropout_rate=rate))

# Convert all results to a DataFrame and save to CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv('ablation_study_results.csv', index=False)
print("\nAll results saved to 'ablation_study_results.csv'")
