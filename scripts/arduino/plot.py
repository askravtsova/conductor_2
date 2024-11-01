"""
File: plot.py
Author: Anna Kravtsova written for Thesis (2024)
Date: September, 2024
Description: plot version x for angulr velocity in the x domain
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Define paths for IMU sample files (replace with your actual file paths)
imu_files = {
    'sharp': '44_90_sharp.csv',
    'normal': '44_90_normal.csv',
    'smooth': '44_90_smooth.csv'
}

# Function to calculate smoothness (standard deviation of acceleration)
def calculate_smoothness(acceleration_data):
    return np.std(acceleration_data)

# Parameters for peak detection (adjust based on data characteristics)
peak_detection_params = {
    'height': 0.2,  # Example height threshold
    'distance': 5  # Minimum distance between peaks
}

# Create a figure for the subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
fig.suptitle("IMU Data with Detected Peaks for Sharp, Normal, and Smooth Gestures", fontsize=16)

# Process each IMU file and plot data with peak markers
for i, (gesture_type, file_name) in enumerate(imu_files.items()):
    file_path = os.path.join('scripts/arduino/data_hybrid/', file_name)
    data = pd.read_csv(file_path)
    
    # Detect peaks and calculate smoothness for Angular Velocity X
    peaks_av, _ = find_peaks(data['Angular_Velocity_X'], height=peak_detection_params['height'], 
                             distance=peak_detection_params['distance'])
    smoothness_av = calculate_smoothness(data['Angular_Velocity_X'])
    
    # Plot Angular Velocity X with detected peaks
    axes[i].plot(data['Time(ms)'], data['Angular_Velocity_X'], label='Angular_Velocity_X', color='navy')
    axes[i].plot(data['Time(ms)'].iloc[peaks_av], data['Angular_Velocity_X'].iloc[peaks_av], "x", color="red", label="Detected Peaks")
    
    # Add plot details
    axes[i].set_title(f"{gesture_type.capitalize()} Gesture - Smoothness: {smoothness_av:.2f}, Peaks: {len(peaks_av)}")
    axes[i].set_ylabel("Angular Velocity X")
    axes[i].legend(loc='upper right')

# Set x-axis label on the last subplot
axes[-1].set_xlabel("Time (ms)")

# Adjust layout and show the plot
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
