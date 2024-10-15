# train_lstm.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import pandas as pd

# Load preprocessed data
X_train = np.load('X_train.npy')
X_val = np.load('X_val.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')
y_test = np.load('y_test.npy')

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
print(f"LSTM Test Accuracy: {test_acc * 100:.2f}%")

# Save history
hist_df = pd.DataFrame(history.history)
hist_csv_file = 'lstm_model_history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
