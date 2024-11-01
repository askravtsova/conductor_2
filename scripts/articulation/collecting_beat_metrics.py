"""
File: collecting beat metrics.py
Author: Anna Kravtsova written for Thesis (2024)
Date: September, 2024
Description: bea bt metrics from data
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import os

# Function to calculjnlate smoothness (standard deviation of acceleration)
def calculate_smoothness(acceleration_data):
    return np.std(acceleration_data)

# Sample file paths 
directory = 'data/articulation_dataset/processed/4_4_90bpm'
samples = {
    'sharp': 'sharp/4_4_90bpm_sharp_sample_1.npy',
    'normal': 'normal/4_4_90bpm_normal_sample_1.npy',
    'smooth': 'smooth/4_4_90bpm_smooth_sample_1.npy'
}

# Parameter sets for testing
parameter_sets = [
    {'height': 0.2, 'prominence': 0.1, 'distance': 7},
    {'height': 0.3, 'prominence': 0.15, 'distance': 15},
    {'height': 0.4, 'prominence': 0.2, 'distance': 30}
]

# Prepare a lisb t to store results for the table
results = []

# Smoothing factor
sigma = 2.0

# Loop over each sample and parameter set
for sample_name, sample_file in samples.items():
    # Load data
    file_path = os.path.join(directory, sample_file)
    data = np.load(file_path, allow_pickle=True)

    # Extract velocity data and calculate acceleration
    velocity_data = [entry[1] for entry in data if entry[1]]
    velocity_data = np.concatenate(velocity_data)
    acceleration_data = np.diff(velocity_data, prepend=velocity_data[0])

    # Smoioth the velocity data
    smoothed_velocity = gaussian_filter1d(velocity_data, sigma=sigma)

    # Calculate smoothness score
    smoothness_score = calculate_smoothness(acceleration_data)

    # Apply each parameter set and count peaks
    for params in parameter_sets:
        peaks, _ = find_peaks(smoothed_velocity, height=params['height'],
                              prominence=params['prominence'], distance=params['distance'])
        
        # Append results to list
        results.append({
            'Sample': sample_name.capitalize(),
            'Height': params['height'],
            'Prominence': params['prominence'],
            'Distance': params['distance'],
            'Number of Peaks': len(peaks),
            'Smoothness Score': smoothness_score
        })

# Create DataFrame for a clean table output
df_results = pd.DataFrame(results)
print(df_results)
