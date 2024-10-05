"""
File: process_data.py
Author: Anna Kravtsova written for Thesis (2024)
Date: Sept, 2024
Description: This script processes each video in raw_vids/ and save the landmarks to a .npy file in processed_data/
also saving the velocity and acceleration data from computing them fro the landmarks

What the accel vel does:
For each frame, the velocity (change in position) and acceleration (change in velocity) for each landmark is calculated
This extends each frame’s data to include landmarks + velocity + acceleration from previous script

"""

import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    landmark_data = []
    
    previous_landmarks = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                current_landmarks = []
                for lm in hand_landmarks.landmark:
                    current_landmarks.append([lm.x, lm.y, lm.z])
                    
        # if there are previous landmarks, calculate the velocity and the acceleration
        if previous_landmarks is not None:
            velocity = []
            acceleration = []
            for i in range(len(current_landmarks)):
                # velocity = difference between current and previous landmarks
                v = np.linalg.norm(np.array(current_landmarks[i]) - np.array(previous_landmarks[i]))
                velocity.append(v) # add on to data
                
                #acceleration = difference between the current and previous velocity
                if len(landmark_data) > 0:
                            previous_velocity = landmark_data[-1][-2][i]  # -2 to get previous velocity
                            a = v - previous_velocity
                            acceleration.append(a)
            
                #append the current landmarks along with velocity and accelration to the data
                landmark_data.append([current_landmarks, velocity, acceleration])
                
                previous_landmarks = current_landmarks
                
    np.save(output_path, np.array(landmark_data))
    cap.release()

if __name__ == "__main__":
    raw_videos_dir = 'data/raw_videos/'
    processed_data_dir = 'data/processed_data_vel_accel/'

    # Traverse through all subdirectories in raw_videos_dir
    for root, dirs, files in os.walk(raw_videos_dir):
        for video_file in files:
            if video_file.endswith('.avi'):
                input_video_path = os.path.join(root, video_file)
                
                # Create a corresponding directory in processed_data_dir
                relative_path = os.path.relpath(root, raw_videos_dir)
                output_dir = os.path.join(processed_data_dir, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                # Create the output path for the .npy file
                output_data_path = os.path.join(output_dir, video_file.replace('.avi', '.npy'))
                
                print(f"Processing {video_file} from {relative_path}")
                process_video(input_video_path, output_data_path)
                print(f"Saved processed data to {output_data_path}")
