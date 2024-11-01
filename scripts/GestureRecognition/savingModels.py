import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Dense, Dropout, Flatten

# model functions

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

# Best hyperparameters from tuning results
best_hyperparameters = {
    "CNN": {"learning_rate": 0.001, "batch_size": 16, "dropout_rate": 0.5},
    "CNN-LSTM": {"learning_rate": 0.001, "batch_size": 16, "dropout_rate": 0.5}
}

# Directory to save models
save_dir = "saved_models/"
os.makedirs(save_dir, exist_ok=True)

# Load and preprocess data (adjust function as needed)
def load_all_npy_files(directory, max_seq_len=100):
    X, y = [], []
    for root, _, files in os.walk(directory):
        for npy_file in files:
            if npy_file.endswith('.npy'):
                file_path = os.path.join(root, npy_file)
                landmarks = np.load(file_path)

                if len(landmarks) > max_seq_len:
                    landmarks = landmarks[:max_seq_len]
                else:
                    landmarks = np.pad(landmarks, ((0, max_seq_len - len(landmarks)), (0, 0)), 'constant')
                
                landmarks = landmarks.reshape(max_seq_len, -1)
                X.append(landmarks)
                # Set labels based on folder names (adjust as needed)
                label = os.path.basename(root)
                y.append({"2_4_60bpm": 0, "3_4_60bpm": 1, "4_4_60bpm": 2}.get(label, -1))
    return np.array(X), np.array(y)

# Load your main dataset
processed_data_dir = 'data/processed_data'
X, y = load_all_npy_files(processed_data_dir)

# Train and save each model with optimal hyperparameters
for model_name, model_fn in {"CNN": create_cnn_model, "CNN-LSTM": create_cnn_lstm_model}.items():
    params = best_hyperparameters[model_name]
    model = model_fn((X.shape[1], X.shape[2]), params['dropout_rate'])
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=30, batch_size=params['batch_size'], validation_data=(X_val, y_val), callbacks=[early_stopping])

    model.save(f"{save_dir}{model_name}_model.h5")
    print(f"Saved {model_name} model at {save_dir}{model_name}_model.h5")

