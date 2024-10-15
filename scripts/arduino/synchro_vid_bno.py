"""

File: synchro_vid_bno.py
Author: Anna Kravtsova written for Thesis (2024)
Date: Oct, 2024
Description: This script takes bno055 data and synchronises 
it with video recorded on device for comaprison later

"""
import serial
import cv2
import time
import os

# Set the folder path
folder_path = 'data_hybrid/test1'

# Check if the folder exists, if not, create it
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Set the file paths for the video and CSV
video_file_path = os.path.join(folder_path, 'output_video.avi')
csv_file_path = os.path.join(folder_path, 'sensor_data.csv')

# Initialize serial communication with the Arduino
ser = serial.Serial('COM3', 9600)  # Replace 'COM3' with the correct port

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use the default camera
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define the codec
out = cv2.VideoWriter(video_file_path, fourcc, 20.0, (640, 480))  # Save video to the file in the folder

start_time = time.time()
duration = 10  # Record for 10 seconds

# Open the CSV file to save sensor data
with open(csv_file_path, 'w') as f:
    # Write CSV header
    f.write('Time(ms),Orientation_X,Orientation_Y,Orientation_Z,Angular_Velocity_X,Angular_Velocity_Y,Angular_Velocity_Z,Linear_Acceleration_X,Linear_Acceleration_Y,Linear_Acceleration_Z\n')

    while (time.time() - start_time) < duration:
        # Capture video frame
        ret, frame = cap.read()
        if ret:
            # Write the video frame to the output file
            out.write(frame)

        # Read sensor data from Arduino
        if ser.in_waiting > 0:
            sensor_data = ser.readline().decode('utf-8').strip()  # Get the sensor data
            current_time = time.time() - start_time  # Get the current time stamp
            # Save sensor data with timestamp
            f.write(f"{current_time},{sensor_data}\n")
        
        # Display the video frame (optional)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Allow user to quit early by pressing 'q'
            break

# Release the video capture and file resources
cap.release()
out.release()
cv2.destroyAllWindows()
ser.close()
