import numpy as np
import os
import json
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

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

                print(f"Loaded {npy_file} with shape: {landmarks.shape}")  # Debugging info

                # Pad or truncate sequences to max_seq_len
                if len(landmarks) > max_seq_len:
                    landmarks = landmarks[:max_seq_len]  # Truncate longer sequences
                else:
                    # Apply padding to the first dimension (timesteps) only, leaving landmarks and coordinates untouched
                    landmarks = np.pad(landmarks, ((0, max_seq_len - len(landmarks)), (0, 0), (0, 0)), 'constant')

                print(f"After padding, {npy_file} has shape: {landmarks.shape}")  # Debugging info

                # Reshape landmarks to (max_seq_len, num_landmarks * 3)
                landmarks = landmarks.reshape(max_seq_len, -1)

                # Labeling based on folder names
                label = os.path.basename(root)
                if label == "2_4_90bpm":
                    y.append(0)  # Label for 2/4 time signature
                elif label == "3_4_90bpm":
                    y.append(1)  # Label for 3/4 time signature
                elif label == "4_4_90bpm":
                    y.append(2)  # Label for 4/4 time signature

                X.append(landmarks)

    return np.array(X), np.array(y)

# Define the paths for your processed data
processed_data_dir = 'data/bigger_dataset/processed_data'

# Load all the data
X, y = load_all_npy_files(processed_data_dir, max_seq_len=100)

# Split the data into train, validation, and test sets (70% train, 15% val, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
model.add(Dropout(0.5))  # Dropout to prevent overfitting
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))  # Assuming 3 classes: 2/4, 3/4, 4/4

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=60, batch_size=16)

# Evaluate on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save history to a CSV file
hist_df = pd.DataFrame(history.history)
hist_csv_file = 'model_trained_data_14_Oct_bigger_dataset.csv'
hist_df.to_csv(hist_csv_file, index=False)
