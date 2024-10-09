import cv2
import os
import time
import tkinter as tk
from tkinter import messagebox
import mediapipe as mp

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to capture and annotate video with hand landmarks (finger pose)
def record_video(output_path, duration=10, update_gui_callback=None):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Unable to read camera feed")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

    start_time = time.time()

    while int(time.time() - start_time) < duration:
        ret, frame = cap.read()

        if ret:
            # Convert the frame to RGB for MediaPipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # If hand landmarks are detected, draw them on the frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Write the frame (with landmarks) to the output file
            out.write(frame)

            if update_gui_callback:
                update_gui_callback("Recording in progress...")

            # Display the frame (optional, you can remove this if you don't need to display)
            cv2.imshow('Recording', frame)

            # Press Q to stop recording early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            print("Error capturing video frame.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Function to record samples and update GUI with details
def record_samples(time_signature, bpm, articulation, sample_count, duration=10, update_gui_callback=None):
    output_dir = f'data/GUI/artistic/{articulation}/{time_signature}_{bpm}bpm'
    os.makedirs(output_dir, exist_ok=True)

    for i in range(sample_count):
        output_filename = f"{time_signature}_{bpm}bpm_sample_{i+1}.avi"
        output_path = os.path.join(output_dir, output_filename)

        if update_gui_callback:
            update_gui_callback(f"Recording {articulation} sample {i+1}/{sample_count} for {time_signature} at {bpm} BPM")

        record_video(output_path, duration=duration, update_gui_callback=update_gui_callback)

        if update_gui_callback:
            update_gui_callback(f"Saved recording to {output_path}")
        
        if i < sample_count - 1:
            time.sleep(2)

# Function to start the recording process from GUI
def start_recording():
    time_signature = time_signature_entry.get()
    bpm = bpm_entry.get()
    articulation = articulation_entry.get()
    sample_count = int(sample_count_entry.get())
    duration = int(duration_entry.get())

    if not time_signature or not bpm or sample_count <= 0 or duration <= 0:
        messagebox.showerror("Error", "Please enter valid details.")
        return

    # Update status label
    def update_status(status_text):
        status_label.config(text=status_text)

    record_samples(time_signature, bpm, articulation, sample_count, duration, update_gui_callback=update_status)

# Set up the GUI
root = tk.Tk()
root.title("Gesture Recording System")
root.geometry("400x300")

# Define and place GUI elements
tk.Label(root, text="Time Signature:").grid(row=0, column=0)
time_signature_entry = tk.Entry(root)
time_signature_entry.grid(row=0, column=1)
time_signature_entry.insert(0, "4_4")

tk.Label(root, text="BPM:").grid(row=1, column=0)
bpm_entry = tk.Entry(root)
bpm_entry.grid(row=1, column=1)
bpm_entry.insert(0, "60")

tk.Label(root, text="Articulation:").grid(row=2, column=0)
articulation_entry = tk.Entry(root)
articulation_entry.grid(row=2, column=1)
articulation_entry.insert(0, "Sharp-staccato")

tk.Label(root, text="Sample Count:").grid(row=3, column=0)
sample_count_entry = tk.Entry(root)
sample_count_entry.grid(row=3, column=1)
sample_count_entry.insert(0, "5")

tk.Label(root, text="Duration (seconds):").grid(row=4, column=0)
duration_entry = tk.Entry(root)
duration_entry.grid(row=4, column=1)
duration_entry.insert(0, "10")

start_button = tk.Button(root, text="Start Recording", command=start_recording)
start_button.grid(row=5, column=0, columnspan=2)

status_label = tk.Label(root, text="Status: Waiting for input...")
status_label.grid(row=6, column=0, columnspan=2)

# Start the GUI loop
root.mainloop()

