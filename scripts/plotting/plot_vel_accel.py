import matplotlib.pyplot as plt

# Assuming `features` now has your combined position, velocity, and acceleration data

# Plot Velocity vs Acceleration for each class (gesture)
for label in unique_gesture_labels:  # Assuming you have gesture labels
    gesture_data = features[labels == label]  # Filter data for this gesture
    plt.scatter(gesture_data['velocity_x'], gesture_data['acceleration_x'], label=f"Gesture {label}")

plt.xlabel('Velocity')
plt.ylabel('Acceleration')
plt.legend()
plt.title('Velocity vs Acceleration for Different Gestures')
plt.show()
