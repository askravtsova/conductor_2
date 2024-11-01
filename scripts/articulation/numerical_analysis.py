"""
File: muerical analysis.py
Author: Anna Kravtsova written for Thesis (2024)
Date: September, 2024
Description: numerical analysis if the velocity data
"""
import numpy as np
import os

def extract_features(directory):
    avg_velocities = []
    var_velocities = []
    max_velocities = []
    
    avg_accelerations = []
    var_accelerations = []
    max_accelerations = []
    
    # Load all .npy files in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                data = np.load(file_path, allow_pickle=True)
                
                # Extract velocities and accelerations
                velocities = [entry[1] for entry in data]
                accelerations = [entry[2] for entry in data]
                
                # Flatten the velocity and acceleration lists
                flat_velocities = np.concatenate(velocities)
                flat_accelerations = np.concatenate(accelerations)
                
                # Feature extraction: mean, variance, max for velocity and acceleration
                avg_velocities.append(np.mean(flat_velocities))
                var_velocities.append(np.var(flat_velocities))
                max_velocities.append(np.max(flat_velocities))
                
                avg_accelerations.append(np.mean(flat_accelerations))
                var_accelerations.append(np.var(flat_accelerations))
                max_accelerations.append(np.max(flat_accelerations))
    
    return (avg_velocities, var_velocities, max_velocities, 
            avg_accelerations, var_accelerations, max_accelerations)

# for 2/4 time signature
directory = 'data/processed_data_vel_accel/2_4_60bpm'
avg_velocities, var_velocities, max_velocities, avg_accelerations, var_accelerations, max_accelerations = extract_features(directory)

print("Average Velocities:", avg_velocities)
print("Variance in Velocities:", var_velocities)
print("Max Velocities:", max_velocities)
print("Average Accelerations:", avg_accelerations)
print("Variance in Accelerations:", var_accelerations)
print("Max Accelerations:", max_accelerations)