"""
File: process_data.py
Author: Anna Kravtsova written for Thesis (2024)
Date: September, 2024
Description: This script is used for training the model, this is the first attempt. 
also splitting the data set for both train validation and testing sets
"""
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Function to load all .npy files from the processed data directories
def load_all_npy_files(processed_data_dir):
    X = []  # List to store landmark data
    y = []  # List to store corresponding labels
    
    # Traverse through all subdirectories in processed_data_dir
    for root, dirs, files in os.walk(processed_data_dir):
        for npy_file in files:
            if npy_file.endswith('.npy'):
                file_path = os.path.join(root, npy_file)
                # Load the landmarks data from .npy file
                landmarks = np.load(file_path)

                # Assuming the label is part of the folder name (e.g., 2_4, 3_4, 4_4)
                label = os.path.basename(root)  # The folder name is used as the label
                if label == "2_4":
                    y.append(0)  # Label for 2/4 time signature
                elif label == "3_4":
                    y.append(1)  # Label for 3/4 time signature
                elif label == "4_4":
                    y.append(2)  # Label for 4/4 time signature

                # Append the data and label
                X.append(landmarks)

    return np.array(X), np.array(y)

# Define the paths for your processed data
processed_data_dir = 'data/processed_data/'

# Load all the data
X, y = load_all_npy_files(processed_data_dir)

# Split the data into train, validation, and test sets (70% train, 15% val, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Build an LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
model.add(Dropout(0.5))  # Dropout to prevent overfitting
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))  # Assuming 3 classes: 2/4, 3/4, 4/4

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=16)

# Evaluate on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
