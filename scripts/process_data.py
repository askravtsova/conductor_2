"""
File: process_data.py
Author: Anna Kravtsova written for Thesis (2024)
Date: Aug, 2024
Description: This script processes each video in raw_vids/ and save the landmarks to a .npy file in processed_data/
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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
                landmark_data.append(landmarks)

    np.save(output_path, np.array(landmark_data))
    cap.release()

if __name__ == "__main__":
    raw_videos_dir = 'data/raw_videos/'
    processed_data_dir = 'data/processed_data/'

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
