"""
File: peakSP analysis.py
Author: Anna Kravtsova written for Thesis (2024)
Date: September, 2024
Description: analyse all sharp norm and smooth getsures and plot
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def load_articulation_data(directory):
    velocities = []
    accelerations = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                data = np.load(file_path, allow_pickle=True)
                
                # Extract velocities and accelerations
                for entry in data:
                    velocities.append(entry[1])    # Collecbj t velocity data
                    accelerations.append(entry[2])   # Collect acceleration data

    # Flatten lists ofmn  lists into single arrays
    velocities = np.concatenate(velocities)
    accelerations = np.concatenate(accelerations)
    
    return velocities, accelerations

def estimate_bpm(velocity_data, fps=30, height=0.02, prominence=0.01, sigma=1):
    # Apply Gaussian smoothinng with adjustable sigma
    smoothed_velocity = gaussian_filter1d(velocity_data, sigma=sigma)
    
    # Detect peaks with specified height and prominence
    peaks, _ = find_peaks(smoothed_velocity, height=height, distance=fps // 2, prominence=prominence)

    # Calculate intervals between peaks and estimate BPM
    if len(peaks) > 1:
        intervals = np.diff(peaks) / fps
        avg_interval = np.mean(intervals)
        bpm = 60 / avg_interval if avg_interval > 0 else 0
    else:
        bpm = 0  # Insufficient peaks to calculate BPM

    return bpm, peaks

# Function to calculate smoothness (std dev of ac celeration)
def calculate_smoothness(acceleration_data):
    return np.std(acceleration_data)

# Directories for each articulation type
sharp_dir = 'data/articulation_dataset/processed/4_4_60bpm/sharp'
normal_dir = 'data/articulation_dataset/processed/4_4_60bpm/normal'
smooth_dir = 'data/articulation_dataset/processed/4_4_60bpm/smooth'

# Load data
velocities_sharp, accelerations_sharp = load_articulation_data(sharp_dir)
velocities_normal, accelerations_normal = load_articulation_data(normal_dir)
velocities_smooth, accelerations_smooth = load_articulation_data(smooth_dir)

# ach articulation type with custom parameters
bpm_sharp, peaks_sharp = estimate_bpm(velocities_sharp, height=0.02, prominence=0.01, sigma=0.5)
smoothness_sharp = calculate_smoothness(accelerations_sharp)
print(f"Sharp Articulation - Estimated BPM: {bpm_sharp:.2f}, Smoothness: {smoothness_sharp:.2f}")

bpm_normal, peaks_normal = estimate_bpm(velocities_normal, height=0.015, prominence=0.008, sigma=1)
smoothness_normal = calculate_smoothness(accelerations_normal)
print(f"Normal Articulation - Estimated BPM: {bpm_normal:.2f}, Smoothness: {smoothness_normal:.2f}")

bpm_smooth, peaks_smooth = estimate_bpm(velocities_smooth, height=0.01, prominence=0.005, sigma=1.5)
smoothness_smooth = calculate_smoothness(accelerations_smooth)
print(f"Smooth Articulation - Estimated BPM: {bpm_smooth:.2f}, Smoothness: {smoothness_smooth:.2f}")

# Plot 
plt.figure(figsize=(8, 4))
plt.plot(velocities_sharp, label="Velocity")
plt.plot(peaks_sharp, velocities_sharp[peaks_sharp], "x", label="Detected Peaks")
plt.xlabel("Frame")
plt.ylabel("Velocity")
plt.legend()
plt.title("Velocity Peaks - Sharp Articulation")
plt.show()
