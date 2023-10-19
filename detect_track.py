from collections import defaultdict
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gc
from movenet_helper import *
gc.enable()

# Function to extract roi
def extract_roi(bbox, frame):
    x1, y1, x2, y2 = [int(coord) for coord in bbox]

    # Ensure the bbox is within the frame boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(x2, frame.shape[1])
    y2 = min(y2, frame.shape[0])

    if x2 <= x1 or y2 <= y1:
        print(f"Invalid bounding box coordinates: x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}")
        return None

    roi = frame[y1:y2, x1:x2]

    gc.collect()

    return roi

def detect_track(frame, model, debug=True):        
    # Store the track history and frames list
    track_history = defaultdict(list)
    
    # Run YOLOV8 Tracking
    results = model.track(frame, classes=[0], persist=True, ) # Run detection and tracking | tracker="bytetrack.yaml"

    for result in results:
        if result.boxes is None or result.boxes.id is None:
            continue
        else:       
            # Gettign class id for each labels -> persons : 0
            class_indices = result.boxes.cls.numpy().astype(int)
            boxes = result.boxes.xyxy.numpy().astype(int)
            ids = result.boxes.id.numpy().astype(int)
            
            # For each 'person' bounding box, draw it on the frame
            for box, person_id in zip(boxes, ids):
                # Extract ROI
                if debug: print('Extract rois')
                roi = extract_roi(box, frame)
                image_height, image_width, _ = roi.shape
                crop_region = init_crop_region(image_height, image_width)

                # Run Movenet model interference
                if debug: print('Extract keypoints using movenet')
                keypoints_with_scores = run_inference(movenet, roi, crop_region, [input_size, input_size])            

                # Append for tracking
                track_history[person_id].append((roi, box, keypoints_with_scores))
    
    return track_history