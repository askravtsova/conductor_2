"""
File: record_gestures.py
Author: Anna Kravtsova written for Thesis (2024)
Date: Aug, 2024
Description: This script processes each video from camera depending on how many videos highlighted to be 
saved:
# Define the time signature and BPM for this collection
    time_signature = '4_4'
    bpm = '60'
    sample_count = 5
    duration = 10  # Duration of each recording in seconds
saving the details to a folder with this many recoridngs = (sample count)
"""

import cv2
import os
import time

def record_video(output_path, duration=10):
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened(): 
        print("Unable to read camera feed")
        return

    #frame_width = int(cap.get(3))
    #frame_height = int(cap.get(4))
    # Get the video writer initialized to save the output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object.
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
    
    start_time = time.time()
    frame_count = 0

    while int(time.time() - start_time) < duration:
        ret, frame = cap.read()

        if ret:
            # Write the frame into the file
            out.write(frame)

            # Display the resulting frame    
            cv2.imshow('frame', frame)

            # Press Q on keyboard to stop recording early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Error capturing video frame.")
            break 

    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def record_samples(time_signature, bpm, sample_count, duration=10):
    # Ensure the output directory exists
    output_dir = f'data/raw_videos/{time_signature}_{bpm}bpm'
    os.makedirs(output_dir, exist_ok=True)

    for i in range(sample_count):
        # Create a unique filename for each recording
        output_filename = f"{time_signature}_{bpm}bpm_sample_{i+1}.avi"
        output_path = os.path.join(output_dir, output_filename)

        print(f"Recording sample {i+1}/{sample_count} for {time_signature} at {bpm} BPM")
        record_video(output_path, duration=duration)
        print(f"Saved recording to {output_path}")

        # Optional: Wait for a short time before starting the next recording
        if i < sample_count - 1:
            print("Get ready for the next sample...")
            time.sleep(2)  # Adjust the wait time as needed

if __name__ == "__main__":
    # Define the time signature and BPM for this collection
    time_signature = '4_4'
    bpm = '60'
    sample_count = 30
    duration = 10  # Duration of each recording in seconds

    # Start recording samples
    record_samples(time_signature, bpm, sample_count, duration=duration)

