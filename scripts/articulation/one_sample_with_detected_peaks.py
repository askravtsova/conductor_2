"""
File: one_sample_with_detected_peaks.py modified for one sample
Author: Anna Kravtsova written for Thesis (2024)
Date: Sept, 2024
Description: This script is for beat and articulation analysis 

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import os

# Function to calculate BPM based on detected peaks in velocity data
def estimate_bpm(peaks, fps=15):
    if len(peaks) > 1:
        intervals = np.diff(peaks) / fps  # Intervals in seconds
        avg_interval = np.mean(intervals)
        bpm = 60 / avg_interval if avg_interval > 0 else 0  # Convert interval to BPM
    else:
        bpm = 0  # Not enough peaks to calculate BPM
    return bpm

# Load data
directory = 'data/articulation_dataset/processed/4_4_60bpm/sharp'
sample_file = '4_4_60bpm_sharp_sample_4.npy'  # Replace with an actual sample file from your dataset
file_path = os.path.join(directory, sample_file)
data = np.load(file_path, allow_pickle=True)

# Extract velocity data and flatten it
velocity_data = [entry[1] for entry in data if entry[1]]
velocity_data = np.concatenate(velocity_data)

# Parameters for peak detection
fps = 15
sigma = 2.0  # Smooth the velocity data
parameter_sets = [
    {'height': 0.2, 'prominence': 0.1, 'distance': fps // 2},
    {'height': 0.3, 'prominence': 0.15, 'distance': fps},
    {'height': 0.4, 'prominence': 0.2, 'distance': fps * 2}
]

# Smooth velocity data
smoothed_velocity = gaussian_filter1d(velocity_data, sigma=sigma)

# Set up plot
fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

for i, params in enumerate(parameter_sets):
    # Detect peaks
    peaks, _ = find_peaks(smoothed_velocity, height=params['height'], prominence=params['prominence'], distance=params['distance'])
    bpm = estimate_bpm(peaks, fps=fps)

    # Plot
    axes[i].plot(velocity_data, color='gray', label='Raw Velocity', alpha=0.5)
    axes[i].plot(smoothed_velocity, color='blue', label='Smoothed Velocity')
    axes[i].plot(peaks, smoothed_velocity[peaks], "x", color='red', label="Detected Peaks")
    axes[i].set_title(f"Height={params['height']}, Prominence={params['prominence']}, Distance={params['distance']}")
    axes[i].set_ylabel("Velocity")
    axes[i].legend()

axes[-1].set_xlabel("Frame")
plt.suptitle("Effect of Different Peak Detection Parameters on Beat Detection", fontsize=16)
plt.show()



