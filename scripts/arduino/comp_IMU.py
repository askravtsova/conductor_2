"""
File: comp_IMU.py
Author: Anna Kravtsova written for Thesis (2024)
Date: September, 2024
Description: getting peaks from imu data
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Define paths for IMU sample files 
imu_files = {
    'Sharp': '44_90_sharp.csv',
    'Normal': '44_90_normal.csv',
    'Smooth': '44_90_smooth.csv'
}

# Define parameter sets to test
parameter_sets = [
    {'height': 0.3, 'prominence': 0.1},
    {'height': 0.5, 'prominence': 0.15},
    {'height': 1, 'prominence': 0.2}
]

# List to store results for table
results = []

# Set up a figure with 3 rows (one for each gesture type)
fig, axes = plt.subplots(len(imu_files), 1, figsize=(12, 18), sharex=True)
fig.suptitle("IMU Data with Detected Peaks for Sharp, Normal, and Smooth Gestures", fontsize=16)

# Process each IMU file and parameter set
for i, (gesture_type, file_name) in enumerate(imu_files.items()):
    file_path = os.path.join('scripts/arduino/data_hybrid/', file_name)
    data = pd.read_csv(file_path)
    
    # Plot configuration for each row
    ax = axes[i]
    ax.plot(data['Time(ms)'], data['Angular_Velocity_X'], label=f'{gesture_type} - Angular Velocity X', color='navy')

    # Loop through parameter sets and detect peaks
    for params in parameter_sets:
        # Detect peaks with current parameter set
        peaks, _ = find_peaks(data['Angular_Velocity_X'], height=params['height'], 
                              prominence=params['prominence'], distance=15)
        
        # Append results to the table
        results.append({
            'Sample': gesture_type,
            'Height': params['height'],
            'Prominence': params['prominence'],
            'Detected Peaks': len(peaks)
        })

        # Plot peaks on the data
        ax.plot(data['Time(ms)'].iloc[peaks], data['Angular_Velocity_X'].iloc[peaks], "x", 
                label=f"Peaks (Height={params['height']}, Prominence={params['prominence']})")

    # Labeling
    ax.set_title(f"{gesture_type} Gesture - Detected Peaks")
    ax.set_ylabel("Angular Velocity X")
    ax.legend()

# Set x-axis label for the last subplot
axes[-1].set_xlabel("Time (ms)")

# Display the plot
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Convert results to DataFrame for display
results_df = pd.DataFrame(results)
print(results_df)




