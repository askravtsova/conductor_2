import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Function to load .npy files with consistent shapes
def load_all_npy_files(processed_data_dir, max_seq_len=100):
    X = []  # List to store landmark data
    y = []  # List to store corresponding labels
    
    for root, dirs, files in os.walk(processed_data_dir):
        for npy_file in files:
            if npy_file.endswith('.npy'):
                file_path = os.path.join(root, npy_file)
                landmarks = np.load(file_path)

                # Check if landmarks have the correct shape
                if landmarks.ndim != 3 or landmarks.shape[1:] != (21, 3):
                    print(f"Warning: File {npy_file} has an unexpected shape {landmarks.shape}")
                    continue  # Skip files with inconsistent shapes

                # Pad or truncate sequences to max_seq_len
                if len(landmarks) > max_seq_len:
                    landmarks = landmarks[:max_seq_len]  # Truncate longer sequences
                else:
                    landmarks = np.pad(landmarks, ((0, max_seq_len - len(landmarks)), (0, 0), (0, 0)), 'constant')  # Pad shorter sequences

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

# Load preprocessed data
processed_data_dir = 'data/bigger_dataset/processed_data'  # Replace with your data directory
X, y = load_all_npy_files(processed_data_dir, max_seq_len=100)

# Check if any data was loaded
if len(X) == 0:
    raise ValueError("No valid data files found. Check for shape inconsistencies.")

# Split the data into train, validation, and test sets (70% train, 15% val, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=60, batch_size=16)

# Evaluate on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
val_loss, val_acc = model.evaluate(X_val, y_val)

# Print accuracy results
print(f"LSTM Validation Accuracy: {val_acc * 100:.2f}%")
print(f"LSTM Test Accuracy: {test_acc * 100:.2f}%")

# Save model history to a CSV file
hist_df = pd.DataFrame(history.history)
hist_df.to_csv('bigger_dataset_lstm_model_history_all_metrics.csv', index=False)

# Step 1: Predict the test labels
y_test_pred = np.argmax(model.predict(X_test), axis=1)

# Step 2: Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)

# Step 3: Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["2/4", "3/4", "4/4"], yticklabels=["2/4", "3/4", "4/4"])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix_lstm.png')
plt.show()

# Step 4: Calculate TP, FP, FN, TN for each class
TP = np.diag(conf_matrix)  # True Positives: Diagonal elements
FP = conf_matrix.sum(axis=0) - TP  # False Positives: Column sum minus diagonal
FN = conf_matrix.sum(axis=1) - TP  # False Negatives: Row sum minus diagonal
TN = conf_matrix.sum() - (FP + FN + TP)  # True Negatives: Total sum minus the rest

# Print the TP, FP, FN, TN for each class
for i, class_name in enumerate(["2/4", "3/4", "4/4"]):
    print(f"Class {class_name}:")
    print(f"  TP: {TP[i]}")
    print(f"  FP: {FP[i]}")
    print(f"  FN: {FN[i]}")
    print(f"  TN: {TN[i]}")

# Step 5: Save confusion matrix results and metrics to CSV
metrics_data = {
    'Class': ["2/4", "3/4", "4/4"],
    'TP': TP,
    'FP': FP,
    'FN': FN,
    'TN': TN
}
metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv('bigger_datset_lstm_confusion_metrics.csv', index=False)

# Step 6: Print classification report (optional)
print("\nClassification Report:\n")
print(classification_report(y_test, y_test_pred, target_names=["2/4", "3/4", "4/4"]))