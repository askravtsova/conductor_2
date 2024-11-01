"""
File: ariculation plot.py
Author: Anna Kravtsova written for Thesis (2024)
Date: September, 2024
Description: plotting the different articulation types on bar chart
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the base directory where your data files are stored
base_dir = 'scripts/arduino/data_artic'

# List of gesture types and sample files
gesture_types = ['44_90_sharp', '44_90_normal', '44_90_smooth']
sample_files = ['s1.csv', 's2.csv', 's3.csv', 's4.csv', 's5.csv']

# Function to calculate smoothness (standard deviation of acceleration)
def calculate_smoothness(acceleration_data):
    return np.std(acceleration_data)

# Dictionary to store smoothness scores
smoothness_scores = {gesture: [] for gesture in gesture_types}

# Process each file in the directory structure
for gesture_type in gesture_types:
    for sample_file in sample_files:
        file_path = os.path.join(base_dir, gesture_type, sample_file)
        
        # Read the data from each sample file
        data = pd.read_csv(file_path)
        
        # Calculate smoothness for Angular Velocity X
        smoothness = calculate_smoothness(data['Angular_Velocity_X'])
        
        # Store the smoothness score for the gesture type
        smoothness_scores[gesture_type].append(smoothness)

# Create a bar plot with 5 samples per gesture type
num_samples = len(sample_files)

# Set up figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Define colours for each gesture type
colours = {'44_90_sharp': '#0f4eba', '44_90_normal': '#4a90e2', '44_90_smooth': '#aec7e8'}


# Plot each sample as a separate bar within the gesture type group
for i, gesture_type in enumerate(gesture_types):
    # Offset x positions for each sample in the group
    x_positions = np.arange(num_samples) + i * (num_samples + 1)
    
    # Plot smoothness scores as bars
    ax.bar(x_positions, smoothness_scores[gesture_type], color=colours[gesture_type], label=gesture_type.replace('44_60_', '').capitalize())

# Labeling and formatting
ax.set_xticks([(i * (num_samples + 1)) + 2 for i in range(len(gesture_types))])
ax.set_xticklabels([gt.replace('44_60_', '').capitalize() for gt in gesture_types])
ax.set_xlabel("Gesture Type")
ax.set_ylabel("Smoothness Score (Std Dev of Angular Velocity X)")
ax.set_title("Smoothness Scores for Individual Samples by Gesture Type")
ax.legend(title="Gesture Type")

plt.tight_layout()
plt.show()
