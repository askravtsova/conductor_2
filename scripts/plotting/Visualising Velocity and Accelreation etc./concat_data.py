#concatinate and plot the velocity vs acceleration

import numpy as np
import os
import matplotlib.pyplot as plt

def load_all_npy_files(directory):
    all_landmarks = []
    all_velocities = []
    all_accelerations = []
    
    # Load all .npy files in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                data = np.load(file_path, allow_pickle=True)  # Allow pickle to load object arrays
                
                # Assume data structure: [landmarks, velocity, acceleration]
                for entry in data:
                    all_landmarks.append(entry[0])       # Collect landmarks
                    all_velocities.append(entry[1])      # Collect velocity
                    all_accelerations.append(entry[2])   # Collect acceleration
    
    return all_landmarks, all_velocities, all_accelerations  # Return as lists

# Load all data for a particular gesture (e.g., 2/4 time signature)
directory = 'data/processed_data_vel_accel/4_4_60bpm'
landmarks, velocities, accelerations = load_all_npy_files(directory)

# Check the shapes of the loaded data
print(f"Number of landmarks: {len(landmarks)}")
print(f"Number of velocity samples: {len(velocities)}")
print(f"Number of acceleration samples: {len(accelerations)}")

# Example: Plot velocity vs acceleration (pick first few data points for simplicity)
vel_sample = np.concatenate(velocities[:100])  # Flatten the first 100 velocity entries
accel_sample = np.concatenate(accelerations[:100])  # Flatten the first 100 acceleration entries

plt.figure(figsize=(6, 6))
plt.scatter(vel_sample, accel_sample, c='red', alpha=0.5)
plt.xlabel('Velocity')
plt.ylabel('Acceleration')
plt.title('Velocity vs Acceleration for 4/4 Time Signature')
plt.show()

