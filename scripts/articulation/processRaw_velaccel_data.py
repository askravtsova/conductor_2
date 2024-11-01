"""
File: process_data.py modified for articulation
Author: Anna Kravtsova written for Thesis (2024)
Date: Sept, 2024
Description: This script processes each video in raw_vids/ and save the landmarks to a .npy file in processed_data/
also saving the velocity and acceleration data from computing them fro the landmarks

What the accel vel does:
For each frame, the velocity (change in pon sition) and acceleration (change in velocity) for each landmark is calculated
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
    previous_velocity = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        current_landmarks = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                current_landmarks = []
                for lm in hand_landmarks.landmark:
                    current_landmarks.append([lm.x, lm.y, lm.z])
                    
        if current_landmarks:  # Only process if landmarks were detected
            if previous_landmarks is not None:
                velocity = []
                acceleration = []
                for i in range(len(current_landmarks)):
                    # Calculate velocity k as difference between current and previous landmarks
                    v = np.linalg.norm(np.array(current_landmarks[i]) - np.array(previous_landmarks[i]))
                    velocity.append(v)

                    # If we have previous velocity, calculate acceleration
                    if previous_velocity is not None:
                        a = v - previous_velocity[i]
                        acceleration.append(a)
                    else:
                        acceleration.append(0)  # No acceleration on the first calculation

                # Append the current landmarks, velocity, and acceleration to the data
                landmark_data.append([current_landmarks, velocity, acceleration])
                previous_velocity = velocity  # Update the previous velocity for the next frame
            else:
                # Append just the landmarks for the first frame
                landmark_data.append([current_landmarks, [], []])

            previous_landmarks = current_landmarks  # Update previous landmarks for the next iteration

    # Save the data using `np.save` with `allow_pickle=True` to support nested structures
    np.save(output_path, np.array(landmark_data, dtype=object), allow_pickle=True)
    cap.release()

if __name__ == "__main__":
    raw_videos_dir = 'data/articulation_dataset/raw'
    processed_data_dir = 'data/articulation_dataset/processed_v2'

    # Traverse through all subdirectories in raw_videos_dir
    for root, dirs, files in os.walk(raw_videos_dir):
        for video_file in files:
            if video_file.endswith('.avi'):
                input_video_path = os.path.join(root, video_file)
                
                # Cdir create processed_data_dir
                relative_path = os.path.relpath(root, raw_videos_dir)
                output_dir = os.path.join(processed_data_dir, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                # output path for the .npy file
                output_data_path = os.path.join(output_dir, video_file.replace('.avi', '.npy'))
                
                print(f"Processing {video_file} from {relative_path}")
                process_video(input_video_path, output_data_path)
                print(f"Saved processed data to {output_data_path}")