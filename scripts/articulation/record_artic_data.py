"""
File: record_artic_data.py
Author: Anna Kravtsova written for Thesis (2024)
Date: Aug, 2024
Description: This script processes each video from camera depending on how many videos highlighted to be 
saved, same process as before but for articulation data
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

    # Initialize the video writer to save the output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 15

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width, frame_height))
    
    start_time = time.time()

    while int(time.time() - start_time) < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Error capturing video frame.")
            break 

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def record_samples(time_signature, bpm, sample_count, articulation, duration=10):
    # Create the directory for recordings if it doesn't exist
    output_dir = f'data/articulation_dataset/{time_signature}_{bpm}bpm/{articulation}'
    os.makedirs(output_dir, exist_ok=True)

    for i in range(sample_count):
        # Create a unique filename for each recording
        output_filename = f"{time_signature}_{bpm}bpm_{articulation}_sample_{i+1}.avi"
        output_path = os.path.join(output_dir, output_filename)

        print(f"Recording sample {i+1}/{sample_count} for {time_signature} at {bpm} BPM ({articulation} articulation)")
        record_video(output_path, duration=duration)
        print(f"Saved recording to {output_path}")

        if i < sample_count - 1:
            print("Get ready for the next sample...")
            time.sleep(2)  # Adjust wait time as needed

if __name__ == "__main__":
    # Set parameters here
    time_signature = '4_4'  # Set the time signature (e.g., '4_4')
    bpm = '90'              # Set the BPM (e.g., '60')
    sample_count = 5        # Number of recordings for this articulation type
    duration = 10            # Duration for each recording in seconds
    
    # Set the articulation type here
    selected_articulation = 'smmoth'  # Options: 'sharp', 'smooth', 'normal'

    # Start recording samples for the selected articulation
    record_samples(time_signature, bpm, sample_count, selected_articulation, duration=duration)
