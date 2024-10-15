# preprocess_data.py
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Function to load and preprocess the data
def load_and_preprocess_data(processed_data_dir, max_seq_len=100):
    X = []
    y = []

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

    X = np.array(X)
    y = np.array(y)

    X_flat = X.reshape(X.shape[0], -1)  # Flatten data for non-LSTM models

    # Split into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Save the preprocessed data for each model
    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    np.save('y_test.npy', y_test)

    np.save('X_train_flat.npy', X_flat[:len(X_train)])
    np.save('X_val_flat.npy', X_flat[len(X_train):len(X_train) + len(X_val)])
    np.save('X_test_flat.npy', X_flat[len(X_train) + len(X_val):])

if __name__ == "__main__":
    processed_data_dir = 'data/processed_data/'
    load_and_preprocess_data(processed_data_dir)
