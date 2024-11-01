"""
File: svm train.py
Author: Anna Kravtsova written for Thesis (2024)
Date: September, 2024
Description: flat data svm training
"""
# train_svm.py
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load preprocessed flat data
X_train_flat = np.load('X_train_flat.npy')
X_val_flat = np.load('X_val_flat.npy')
X_test_flat = np.load('X_test_flat.npy')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')
y_test = np.load('y_test.npy')

kernels = ['linear', 'poly', 'rbf']
for kernel in kernels:
    print(f"\nTraining SVM with {kernel} kernel...")
    
    svm_model = SVC(kernel=kernel, random_state=42)
    svm_model.fit(X_train_flat, y_train)
    
    # Evaluate on validation and test sets
    y_val_pred = svm_model.predict(X_val_flat)
    y_test_pred = svm_model.predict(X_test_flat)
    
    # Validation accuracy
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"SVM ({kernel} kernel) Validation Accuracy: {val_acc * 100:.2f}%")
    
    # Test accuracy
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"SVM ({kernel} kernel) Test Accuracy: {test_acc * 100:.2f}%")
    
    # Save classification report
    report = classification_report(y_test, y_test_pred)
    with open(f'svm_{kernel}_classification_report.txt', 'w') as f:
        f.write(report)
