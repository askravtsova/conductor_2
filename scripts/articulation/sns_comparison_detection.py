"""
File: sns comp detectt.py
Author: Anna Kravtsova written for Thesis (2024)
Date: September, 2024
Description: plotting the different articulation types on bar chart
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import os

# Function to calculjb ate BPM based on detected peaks in velocity data
def estimate_bpm(peaks, fps=15):
    if len(peaks) > 1:
        intervals = np.diff(peaks) / fps  # Intervals in seconds
        avg_interval = np.mean(intervals)
        bpm = 60 / avg_interval if avg_interval > 0 else 0  # Convert interval to BPM
    else:
        bpm = 0  # Not enough peaks to calculate BPM
    return bpm

# pavs
directory = 'data/articulation_dataset/processed/4_4_60bpm'
samples = {
    'sharp': 'sharp/4_4_60bpm_sharp_sample_4.npy',   
    'normal': 'normal/4_4_60bpm_normal_sample_3.npy',  
    'smooth': 'smooth/4_4_60bpm_smooth_sample_3.npy'   
}

# Peak detection parameters
fps = 15
sigma = 2.0  # Smooth the velocity data
height = 0.2
prominence = 0.1
distance = fps  # Ensure peaks are separated by at least 1 second

# Set up plot
fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

# Loop over each sample andkhb plot
for i, (gesture_type, sample_file) in enumerate(samples.items()):
    # Load data
    file_path = os.path.join(directory, sample_file)
    data = np.load(file_path, allow_pickle=True)

    # Extract and flatten velocity data
    velocity_data = [entry[1] for entry in data if entry[1]]
    velocity_data = np.concatenate(velocity_data)

    # Smooth velocity data
    smoothed_velocity = gaussian_filter1d(velocity_data, sigma=sigma)

    # Detect peaks
    peaks, _ = find_peaks(smoothed_velocity, height=height, prominence=prominence, distance=distance)
    bpm = estimate_bpm(peaks, fps=fps)

    # Plot
    axes[i].plot(velocity_data, color='gray', label='Raw Velocity', alpha=0.5)
    axes[i].plot(smoothed_velocity, color='blue', label='Smoothed Velocity')
    axes[i].plot(peaks, smoothed_velocity[peaks], "x", color='red', label="Detected Peaks")
    axes[i].set_title(f"{gesture_type.capitalize()} Gesture")
    axes[i].set_ylabel("Velocity")
    axes[i].legend()

axes[-1].set_xlabel("Frame")
plt.suptitle("Comparison of Sharp, Normal, and Smooth Gestures in Beat Detection", fontsize=16)
plt.show()
