"""
File: process_video.py
Author: Anna Kravtsova written for Thesis (2024)
Date: Aug, 2024
Description: This script takes videos from raw_videos/ and appends the landmarks from media 
pipe to visualise the hands and check if any issues occur
"""

import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

#list of landmark indicies for pointer (index) and middle fingers
target_landmarks = [5, 6, 7, 8, 9, 10, 11, 12]
#target_landmarks = [8, 12] # only the tips of these two fingers

def overlay_and_save_landmarks(video_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video opened correctly
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get the video writer initialized to save the output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Define the codec and create VideoWriter object.
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
    
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB as MediaPipe requires RGB input
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = hands.process(frame_rgb)

        # Draw landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for idx in target_landmarks:
                    x = int(hand_landmarks.landmark[idx].x * frame.shape[1])
                    y = int(hand_landmarks.landmark[idx].y * frame.shape[0])
                    cv2.circle(frame, (x,y), 5, (0, 255, 0), -1) #green circles to see pointer and mid finger
                #mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Write the frame with landmarks to the output video
        out.write(frame)

    # Release everything when the job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
video_path = 'data/raw_videos/4_4_60bpm/4_4_60bpm_sample_1.avi'
output_path = 'data/landmark_videos/4_4_60bpm_sample_1_with_targeted_landmarks.avi'
overlay_and_save_landmarks(video_path, output_path)
