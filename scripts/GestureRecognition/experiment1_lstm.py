"""
File: exp lstm.py
Author: Anna Kravtsova written for Thesis (2024)
Date: September, 2024
Description: experiment 1 in expa dnreslts sect
"""
import numpy as np
import os
import json
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Function to load all .npy files from the processed data directories
def load_all_npy_files(processed_data_dir, max_seq_len=100):
    X = []  # List to store landmark data
    y = []  # List to store corresponding labels
    
    # Traverse through all subdirectories in processed_data_dir
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


processed_data_dir = 'data/processed_data'


X, y = load_all_npy_files(processed_data_dir, max_seq_len=100)


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
model.add(Dropout(0.5))  # Dropout to prevent overfitting
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))  # Assuming 3 classes: 2/4, 3/4, 4/4


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=60, batch_size=16)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

y_pred = np.argmax(model.predict(X_test), axis=1)  # Predictions on test set
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["2_4_60bpm", "3_4_60bpm", "4_4_60bpm"]))

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

metrics = {
    "Test Accuracy": test_acc,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}
with open('model_metricsEXPERIMENT1.json', 'w') as f:
    json.dump(metrics, f)

hist_df = pd.DataFrame(history.history)


# Plot training & validation accuracy and loss
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
