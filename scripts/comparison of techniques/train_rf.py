"""
File: trainrf.py
Author: Anna Kravtsova written for Thesis (2024)
Date: September, 2024
Description: train simple rf model base on flattened x vel 
"""
# train_rf.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# Load preprocessed flat data
X_train_flat = np.load('X_train_flat.npy')
X_val_flat = np.load('X_val_flat.npy')
X_test_flat = np.load('X_test_flat.npy')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')
y_test = np.load('y_test.npy')

# Create and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_flat, y_train)

# Evaluate on validation and test set
y_val_pred = rf_model.predict(X_val_flat)
y_test_pred = rf_model.predict(X_test_flat)

# Save and report accuracy
val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Random Forest Validation Accuracy: {val_acc * 100:.2f}%")
print(f"Random Forest Test Accuracy: {test_acc * 100:.2f}%")

# Save classification report to file
report = classification_report(y_test, y_test_pred)
with open('rf_classification_report.txt', 'w') as f:
    f.write(report)
