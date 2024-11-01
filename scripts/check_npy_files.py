"""
File: check npy.py
Author: Anna Kravtsova written for Thesis (2024)
Date: September, 2024
Description: check npy files
"""
import os
import numpy as np

# Define the directory where the processed data is stored
processed_data_dir = 'data/processed_data'  # Update this with your data directory

# Define the expected shape for the landmarks data (for example, 100 frames and 3 coordinates for each frame)
max_seq_len = 100  # Adjust this based on your expected max sequence length
expected_shape = (max_seq_len, 3)

def check_all_npy_shapes(directory, expected_shape):
    inconsistent_files = []  # To keep track of files with unexpected shapes

    for root, dirs, files in os.walk(directory):
        for npy_file in files:
            if npy_file.endswith('.npy'):
                file_path = os.path.join(root, npy_file)
                try:
                    landmarks = np.load(file_path)
                    print(f"Loaded {npy_file} with shape: {landmarks.shape}")
                    
                    # Check if the shape of the file matches the expected shape
                    if landmarks.shape != expected_shape:
                        print(f"Warning: File {npy_file} has an unexpected shape {landmarks.shape}")
                        inconsistent_files.append((npy_file, landmarks.shape))
                except Exception as e:
                    print(f"Error loading file {npy_file}: {e}")
    
    return inconsistent_files

# Run the shape check
inconsistent_files = check_all_npy_shapes(processed_data_dir, expected_shape)

# If any inconsistent files were found, print them out
if inconsistent_files:
    print("\nInconsistent files detected:")
    for file_name, shape in inconsistent_files:
        print(f"File: {file_name}, Shape: {shape}")
else:
    print("\nAll files have consistent shapes.")
