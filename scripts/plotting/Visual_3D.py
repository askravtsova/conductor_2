"""
File: visual3dt.py
Author: Anna Kravtsova written for Thesis (2024)
Date: September, 2024
Description: visual in 3d space
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_gesture_lines(file_path):
    # Load the gesture data from the .npy file
    data = np.load(file_path, allow_pickle=True)
    
    # Create lists to store all the landmarks across frames
    all_xs, all_ys, all_zs = [], [], []

    # Extract landmarks for all frames
    for frame_idx, frame_data in enumerate(data):
        if len(frame_data) > 0:  # Only process if landmarks were detected
            landmarks = np.array(frame_data)  # Convert to numpy array
            xs, ys, zs = landmarks[:, 0], landmarks[:, 1], landmarks[:, 2]
            
            # Store landmarks for the entire gesture
            all_xs.extend(xs)
            all_ys.extend(ys)
            all_zs.extend(zs)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot all landmarks and connect them with very thin lines
    ax.plot(all_xs, all_ys, all_zs, c='b', marker='o', linewidth=0.05, markersize = 2)  # Set linewidth to 0.5 for thin lines

    # Label the axes
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    ax.set_title("Gesture Trajectory in 3D Space")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Path to the specific .npy file
    file_path = 'data/processed_data/4_4_60bpm/4_4_60bpm_sample_1.npy'

    # Visualize the gesture in the specified file
    visualize_gesture_lines(file_path)
