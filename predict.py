from frame_processing import detect_and_track
from movenet_helper import combined_heatmap
from fuse_features import fuse_features
import cv2
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import tensorflow as tf

IMAGE_HEIGHT,IMAGE_WIDTH = 224,224

def extract_frames(video_file):
    capture = cv2.VideoCapture(video_file)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    EXTRACT_FREQUENCY = 3
    if frame_count // EXTRACT_FREQUENCY <= 16:
        EXTRACT_FREQUENCY -= 1
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
    count = 0
    i = 0
    retaining = True
    frames_list = []
    
    while (count < frame_count and retaining):
        retaining, frame = capture.read()
        if frame is None:
            continue
        if count % EXTRACT_FREQUENCY == 0:
            frames_list.append(frame)
            i += 1
        count += 1
        
    capture.release()
    
    return frames_list

def predict_on_video(video_path, 
                     output_video, 
                     CLASSES_LIST,
                     dense_model,
                     lstm_model,
                     cnn_model, 
                     yolo_model, 
                     spatial_resnet, 
                     spatial_transformer, 
                     temporal_resnet, 
                     temporal_transformer, 
                     SEQUENCE_LENGTH, 
                     debug=False):
    """
    Predict the class of a video and save an annotated video with predicted actions.
    """
    feature_buffer = []
    frames_buffer = []
    
    def combined_voting(predictions):
        # Soft voting
        summed = np.sum(predictions, axis=0)
        softmax_probs = tf.nn.softmax(summed).numpy()
        soft_vote = np.argmax(softmax_probs[0])  # Assuming 1 prediction for 1 frame
        
        # Hard voting
        predicted_classes = [np.argmax(pred[0]) for pred in predictions]  # Assuming 1 prediction for 1 frame
        hard_vote = max(set(predicted_classes), key=predicted_classes.count)

        # Debug
        if debug:
            print(f"Soft Vote: {CLASSES_LIST[soft_vote]}")
            print(f"Hard Vote: {CLASSES_LIST[hard_vote]}")
            for i, pred in enumerate(predictions):
                model_name = ["LSTM", "Dense", "CNN"][i]
                print(f"\nModel: {model_name}")
                for idx, class_name in enumerate(CLASSES_LIST):
                    print(f"{class_name}: {pred[0][idx] * 100:.2f}%")
                print('\n')

        # Combination logic:
        # Take hard voting as final if it disagrees with soft voting
        return hard_vote if soft_vote != hard_vote else soft_vote
    
    # Extract frames from the video
    if debug: print("Extracting frames from video...")
    frames = extract_frames(video_path)
    video_dims = (frames[0].shape[1], frames[0].shape[0])
    output_video_path = output_video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, video_dims)

    for frame in frames:
        if debug: print("Processing frame...")
        
        results = list(detect_and_track([frame], yolo_model))
        
        if not results:
            out.write(frame)
            continue

        _, roi_frame, _, _, keypoints_with_scores = results[0]

        # Generate heatmap for the keypoints
        if debug: print("Generating heatmap...")
        heatmap = combined_heatmap(keypoints_with_scores[0, 0], IMAGE_HEIGHT, IMAGE_WIDTH)
        heatmap = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
        heatmap = F.interpolate(heatmap, size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode='bilinear').squeeze().numpy()
        max_value = np.max(heatmap)
        if max_value > 0:  # Avoid division by zero
            heatmap = heatmap / max_value
        heatmap_tensor = torch.tensor(heatmap).float().repeat(1, 3, 1, 1)

        # Process the ROI
        if debug: print("Processing ROI...")
        gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        roi_img = torch.tensor(gray_roi).unsqueeze(0).unsqueeze(0)
        roi_img = F.interpolate(roi_img, size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode='bilinear').squeeze().numpy() 
        roi_img = roi_img / 255.0
        roi_img_tensor = torch.tensor(roi_img).float().repeat(1, 3, 1, 1)
        
        
        # Extract spatial and temporal features
        if debug: print("Extracting spatial and temporal features...")
        with torch.no_grad():
            spatial_features = spatial_resnet(roi_img_tensor).squeeze(-1).squeeze(-1)
            temporal_features = temporal_resnet(heatmap_tensor).squeeze(-1).squeeze(-1)

        # Transform features using transformers
        if debug: print("Transforming features...")
        spatial_features = spatial_transformer(spatial_features)
        temporal_features = temporal_transformer(temporal_features)

        # Combine the features
        if debug: print("Combining features...")
        combined_features = fuse_features(spatial_features.detach().numpy(), temporal_features.detach().numpy())
        feature_buffer.append(combined_features)
        resized_frames = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT)).astype(np.float32) / 255.0 # Normalize to [0, 1]
        frames_buffer.append(resized_frames)
        
        # Make predictions if the buffer is full
        if len(feature_buffer) == SEQUENCE_LENGTH:
            if debug: print("Making prediction...")
            sequence_frames = np.array([frames_buffer]) 
            sequence_features = np.array(feature_buffer).reshape(1, SEQUENCE_LENGTH, -1)

            # Get raw predictions from all models
            raw_predictions_lstm = lstm_model.predict(sequence_features)
            raw_predictions_dense = dense_model.predict(sequence_features)
            raw_predictions_cnn = cnn_model.predict(sequence_frames)

            # Aggregate predictions for voting
            aggregated_predictions = [raw_predictions_lstm, raw_predictions_dense, raw_predictions_cnn]

            # Determine the class index from the averaged predictions
            class_index = combined_voting(aggregated_predictions)
            text = f"Predicted: {CLASSES_LIST[class_index]}"
            print(text)
            cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
            feature_buffer.pop(0)  # Slide the window
            frames_buffer.pop(0)

        out.write(frame)
        
    if 0 < len(feature_buffer) < SEQUENCE_LENGTH:
        if debug: print("Duplicating the last feature to fill the buffer...")
        num_duplicates_needed = SEQUENCE_LENGTH - len(frames_buffer)
        last_frame = frames_buffer[-1]
        duplicated_frames = [last_frame] * num_duplicates_needed
        frames_buffer.extend(duplicated_frames)

        num_duplicates_needed = SEQUENCE_LENGTH - len(feature_buffer)
        last_feature = feature_buffer[-1]
        duplicated_features = [last_feature] * num_duplicates_needed
        feature_buffer.extend(duplicated_features)

        sequence_frames = np.array([frames_buffer]) 
        sequence_features = np.array(feature_buffer).reshape(1, SEQUENCE_LENGTH, -1)
        
        # Get raw predictions from both models
        raw_predictions_lstm = lstm_model.predict(sequence_features)
        raw_predictions_dense = dense_model.predict(sequence_features)
        raw_predictions_cnn = cnn_model.predict(sequence_frames)
        # Aggregate predictions for voting
        aggregated_predictions = [raw_predictions_lstm, raw_predictions_dense, raw_predictions_cnn]

        # Determine the class index from the averaged predictions
        class_index = combined_voting(aggregated_predictions)
    
        text = f"Predicted: {CLASSES_LIST[class_index]}"
        print(text)
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
        out.write(frame)
    out.release()