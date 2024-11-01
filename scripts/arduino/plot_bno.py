"""
File: plotbno.py
Author: Anna Kravtsova written for Thesis (2024)
Date: September, 2024
Description: rough visualisation of bno055 stuff
"""
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
csv_file = 'scripts/arduino/data_hybrid/24_90bpm.csv' 
data = pd.read_csv(csv_file)

# Preview the first few rows of the data
print(data.head())

# Plot orientation data (Euler angles)
plt.figure(figsize=(10, 6))
plt.plot(data['Time(ms)'], data['Orientation_X'], label='Orientation X (Yaw)')
plt.plot(data['Time(ms)'], data['Orientation_Y'], label='Orientation Y (Pitch)')
plt.plot(data['Time(ms)'], data['Orientation_Z'], label='Orientation Z (Roll)')
plt.title('Orientation (Euler Angles) Over Time')
plt.xlabel('Time (ms)')
plt.ylabel('Orientation (Degrees)')
plt.legend()
plt.grid(True)
plt.show()

# Plot angular velocity data
plt.figure(figsize=(10, 6))
plt.plot(data['Time(ms)'], data['Angular_Velocity_X'], label='Angular Velocity X')
plt.plot(data['Time(ms)'], data['Angular_Velocity_Y'], label='Angular Velocity Y')
plt.plot(data['Time(ms)'], data['Angular_Velocity_Z'], label='Angular Velocity Z')
plt.title('Angular Velocity Over Time')
plt.xlabel('Time (ms)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.grid(True)
plt.show()

# Plot linear acceleration data
plt.figure(figsize=(10, 6))
plt.plot(data['Time(ms)'], data['Linear_Acceleration_X'], label='Linear Acceleration X')
plt.plot(data['Time(ms)'], data['Linear_Acceleration_Y'], label='Linear Acceleration Y')
plt.plot(data['Time(ms)'], data['Linear_Acceleration_Z'], label='Linear Acceleration Z')
plt.title('Linear Acceleration Over Time')
plt.xlabel('Time (ms)')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.grid(True)
plt.show()
