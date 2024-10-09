import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_velocity_acceleration_over_time(directory, sample_file):
    # Load the specific .npy file for which you want to visualize the data
    file_path = os.path.join(directory, sample_file)
    data = np.load(file_path, allow_pickle=True)
    
    velocities = [entry[1] for entry in data]
    accelerations = [entry[2] for entry in data]
    
    # Flatten the velocity and acceleration lists
    flat_velocities = np.concatenate(velocities)
    flat_accelerations = np.concatenate(accelerations)
    
    # Plot velocity over time
    plt.figure(figsize=(10, 5))
    plt.plot(flat_velocities, label='Velocity')
    plt.plot(flat_accelerations, label='Acceleration')
    plt.title('Velocity and Acceleration Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# Example: Plot velocity and acceleration over time for one gesture sample
directory = 'data/processed_data_vel_accel/4_4_60bpm'
sample_file = '4_4_60bpm_sample_15.npy'  # Replace with your actual file
plot_velocity_acceleration_over_time(directory, sample_file)
