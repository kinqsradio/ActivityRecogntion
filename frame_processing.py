# Local imports
from movenet_helper import *

# Standard library imports
import os
import cv2
import numpy as np
import gc
gc.enable()


IMAGE_HEIGHT,IMAGE_WIDTH = 224,224

# Function to extract roi
def extract_roi(bbox, frame):
    x1, y1, x2, y2 = [int(coord) for coord in bbox]

    # Ensure the bbox is within the frame boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(x2, frame.shape[1])
    y2 = min(y2, frame.shape[0])

    if x2 <= x1 or y2 <= y1:  # Check if the bbox is valid
        # print(f"Invalid bbox at frame {frame_number}: {bbox}")
        return None

    roi = frame[y1:y2, x1:x2]

    gc.collect()

    return roi

# Select best box (Only seclect the first person that it detects)
def select_best_box(boxes, class_indices, ids, names):
    # Find the index of the smallest ID
    index = np.argmin(ids)
    
    # Check if the class of the box with the smallest ID is 'person'
    if names[class_indices[index]] == 'person':
        return boxes[index]
    return None

def detect_and_track(frames, model):
    names = model.names  # Model's class name

    for frame_number, frame in enumerate(frames):
        try:
            print(f"Processing frame {frame_number}")
            results = model.track(frame, persist=True, tracker="bytetrack.yaml") # Run detection and tracking
            class_indices = results[0].boxes.cls.cpu().numpy().astype(int)  # Get the class indices of the detections
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            if len(boxes) == 0:
                # print(f"No detections were made in frame {frame_number}")
                continue

            # Filter boxes to only contain 'person' detections
            print(f'Filter detections')
            person_boxes = [box for box, class_index in zip(boxes, class_indices) if names[int(class_index)] == 'person']
            if not person_boxes:
                print(f"No 'person' detected in frame {frame_number}")
                continue
            

            print(f'Select best box')
            best_box = select_best_box(boxes, class_indices, ids, names)


            print('Extract ROI')
            # Extract and save ROI
            roi = extract_roi(best_box, frame)

            # Initialize the crop region.
            image_height, image_width, _ = roi.shape
            crop_region = init_crop_region(image_height, image_width)

            # Run model inference.
            print('Extract Keypoints')
            keypoints_with_scores = run_inference(movenet, roi, crop_region, [input_size, input_size])

            # Draw prediction on the frame.
            poses_on_frame = draw_prediction_on_image(roi, keypoints_with_scores, crop_region=None, close_figure=True, output_image_height=image_height)

            # Determine the crop region for the next frame.
            crop_region = determine_crop_region(keypoints_with_scores, image_height, image_width)

            # Draw prediction on a blank image
            blank_image = np.zeros_like(roi)
            skeleton_pose = draw_prediction_on_image(blank_image, keypoints_with_scores, crop_region=None, close_figure=True, output_image_height=image_height)

            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            frame = resized_frame #/ 255.0

            resized_roi = cv2.resize(roi, (IMAGE_HEIGHT, IMAGE_WIDTH))
            roi = resized_roi #/ 255.0

            resized_skeleton_pose = cv2.resize(skeleton_pose, (IMAGE_HEIGHT, IMAGE_WIDTH))
            skeleton_pose = resized_skeleton_pose #/ 255.0

            resized_skeleton_on_frame = cv2.resize(poses_on_frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            poses_on_frame = resized_skeleton_on_frame #/ 255.0

            gc.collect()
            
            yield frame, roi, skeleton_pose, poses_on_frame, keypoints_with_scores
        except Exception as e:
            print(f"An error occurred while processing frame {frame_number}: {str(e)}")
            continue