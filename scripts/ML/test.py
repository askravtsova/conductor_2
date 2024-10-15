"""
File: test.py
Author: Anna Kravtsova written for Thesis (2024)
Date: Aug, 2024
Description: simple write camera feed to video script
"""

import cv2

def get_camera_details():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened(): 
        print("Unable to read camera feed")
        return

    # Get camera properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))

    # Codec as four-character string
    codec_str = chr((codec & 0XFF)) + chr((codec & 0XFF00) >> 8) + chr((codec & 0XFF0000) >> 16) + chr((codec & 0XFF000000) >> 24)

    # Print technical details
    details = {
        "Frame Width": frame_width,
        "Frame Height": frame_height,
        "Frame Rate (FPS)": fps,
        "Codec": codec_str
    }
    
    # Release the camera
    cap.release()

    return details

# Get and display camera details
camera_details = get_camera_details()
print(camera_details)
