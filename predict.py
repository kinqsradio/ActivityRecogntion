from detect_track import detect_track
from movenet_helper import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')


# Intialized some values
IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224

def display_frame(frame):
        # Convert from BGR to RGB for proper display in matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)
        plt.axis('off')  # Hide axis values
        plt.show()


def overlay_heatmap_on_image(image, heatmap, colormap=cv2.COLORMAP_JET, alpha=0.6):
    # Resize heatmap to the size of the image
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    # Convert the heatmap to RGB
    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), colormap)
    # Blend the image and the heatmap
    blended = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return blended

def fuse_features(spatial_features, temporal_features):
    # Combined Features
    return np.concatenate([spatial_features, temporal_features], axis=-1)

def model_predicts(lstm_model, dense_model, cnn_model, CLASSES_LIST, sequence_features, sequence_frames, sequence_rois, debug):
    # Model Prediction
    raw_predictions_lstm = lstm_model.predict(sequence_features)
    raw_predictions_dense = dense_model.predict(sequence_features)
    raw_predictions_cnn = cnn_model.predict(sequence_frames)
    raw_predictions_cnn2 = cnn_model.predict(sequence_rois)

    raw_predictions_cnn_avg = (raw_predictions_cnn + raw_predictions_cnn2) / 2.0

    aggregated_predictions = [raw_predictions_lstm, raw_predictions_dense, raw_predictions_cnn_avg]

    class_index = combined_voting(aggregated_predictions, CLASSES_LIST, debug)
    text = f"{CLASSES_LIST[class_index]}"

    return text

def combined_voting(predictions, CLASSES_LIST, debug):
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
            model_name = ["LSTM" ,"Dense", "CNN"][i]
            # model_name = ["LSTM", "Dense", "CNN"][i]
            print(f"\nModel: {model_name}")
            for idx, class_name in enumerate(CLASSES_LIST):
                print(f"{class_name}: {pred[0][idx] * 100:.2f}%")
        print('\n')

    # Combination logic:
    # Take hard voting as final if it disagrees with soft voting
    return hard_vote if soft_vote != hard_vote else soft_vote

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
                     grad_cam, 
                     SEQUENCE_LENGTH,
                     draw_skeleton, # BOOL
                     draw_bbox, # BOOL
                     debug): #BOOL
    
    feature_buffer = {}
    frames_buffer = {}
    rois_buffer = {}

    cap = cv2.VideoCapture(video_path)

    # Get the video's width, height and frames per second
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 2)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or use 'XVID'
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))


    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        track_history = detect_track(frame, yolo_model)
        for person_id, detections in track_history.items():
            for roi, bbox, keypoints_with_scores in detections:
                if debug: print(f'Processing id: {person_id}')
                x1, y1, x2, y2 = bbox
                
                if debug: print(f'Draw Skeleton for id {person_id}')
                if draw_skeleton:
                    image_height, image_width, _ = roi.shape
                    blank_image = np.zeros_like(roi)
                    roi_poses = draw_prediction_on_image(blank_image, keypoints_with_scores, 
                                                    crop_region=None, close_figure=True, output_image_height=image_height)

                if debug: print(f'Draw  Boxes for id {person_id}')
                if draw_bbox:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a green bounding box

                
                if debug: print(f'Processing Heatmap for id {person_id}')
                heatmap = combined_heatmap(keypoints_with_scores[0, 0], IMAGE_HEIGHT, IMAGE_WIDTH)
                heatmap = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
                heatmap = F.interpolate(heatmap, size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode='bilinear').squeeze().numpy()
                max_value = np.max(heatmap)
                if max_value > 0:  # Avoid division by zero
                    heatmap = heatmap / max_value
                heatmap_tensor = torch.tensor(heatmap).float().repeat(1, 3, 1, 1)
                
                if debug: print(f'Processing ROI for id {person_id}')
                resized_roi = cv2.resize(roi, (IMAGE_HEIGHT, IMAGE_WIDTH))
                normalized_roi = resized_roi.astype(np.float32) / 255.0
                roi_tensor = torch.tensor(normalized_roi).permute(2, 0, 1).unsqueeze(0).float()

                if debug: print(f'Processing frame for id {person_id}')
                resized_frames = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT)).astype(np.float32) / 255.0 # Normalize to [0, 1]

                if debug: print(f'Extracting Features for id {person_id}')
                with torch.no_grad():
                    spatial_features = spatial_resnet(roi_tensor).squeeze(-1).squeeze(-1)
                    temporal_features = temporal_resnet(heatmap_tensor).squeeze(-1).squeeze(-1)

                if debug: print(f"Transforming features for id {person_id}")
                spatial_features = spatial_transformer(spatial_features)
                temporal_features = temporal_transformer(temporal_features)

                if debug: print(f'Fuse features for id {person_id}')
                combined_features = fuse_features(spatial_features.detach().numpy(), temporal_features.detach().numpy())

                if debug: print(f'Appending buffers for id {person_id}')
                if person_id not in feature_buffer:
                    feature_buffer[person_id] = []
                    frames_buffer[person_id] = []
                    rois_buffer[person_id] = []
                feature_buffer[person_id].append(combined_features)
                frames_buffer[person_id].append(resized_frames)
                rois_buffer[person_id].append(normalized_roi)
                if debug:
                    print(f'Id: {person_id}, Feautres buffer: {len(feature_buffer[person_id])}, Frames buffer: {len(feature_buffer[person_id])}, Rois buffer: {len(rois_buffer[person_id])}')

                # START PREDICTION
                for person_id in feature_buffer.keys():
                    if len(feature_buffer[person_id]) == SEQUENCE_LENGTH:
                        if debug: print('Start Prediction Process')
                        if debug: print('Converting Buffers to Numpy')
                        sequence_features = np.array(feature_buffer[person_id]).reshape(1, SEQUENCE_LENGTH, -1)
                        sequence_frames = np.array([frames_buffer[person_id]])
                        sequence_rois = np.array([rois_buffer[person_id]])

                        # Aggregate predictions for voting
                        final_predictions = model_predicts(lstm_model, dense_model, cnn_model, CLASSES_LIST, sequence_features, sequence_frames, sequence_rois, debug)
                        if debug: print(f'Predicted: {final_predictions}')

                        font_scale = 2
                        (text_width, text_height), _ = cv2.getTextSize(final_predictions, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)

                        # Define the starting position of the text: just above the bounding box or at the top of the frame if the bbox is too high.
                        start_x = int((x1 + x2) / 2 - text_width / 2)
                        start_y = y1 - 10  # 10 pixels above the bbox

                        # Check if the text fits in the frame, if not adjust.
                        if start_y - text_height < 0:  
                            start_y = y1 + text_height + 10 
                        
                        # Define the coordinates for the rectangle
                        rect_start_x = start_x
                        rect_start_y = start_y - text_height
                        rect_end_x = start_x + text_width
                        rect_end_y = start_y

                        # Draw the white rectangle
                        # cv2.rectangle(frame, (rect_start_x, rect_start_y), (rect_end_x, rect_end_y), (255, 255, 255), -1)  # -1 means filled rectangle

                        # Draw the text
                        cv2.putText(frame, final_predictions, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)

                        feature_buffer[person_id].pop(0)  # Slide the window
                        frames_buffer[person_id].pop(0)
                        rois_buffer[person_id].pop(0)
            
        #  FRAME DEBUG
        normalized_frame = frame.astype(np.float32) / 255.0
        frame_tensor = torch.tensor(normalized_frame).permute(2, 0, 1).unsqueeze(0).float()
        frame_heatmap = grad_cam.generate_cam(frame_tensor)

        # Attentions frame
        frame_with_attention = overlay_heatmap_on_image(frame, frame_heatmap)
        combined_frame = np.hstack((frame_with_attention, frame))
        out.write(combined_frame)
        if debug:
            # FRAME
            display_frame(combined_frame)
    # Duplicate Last Frame
    for person_id in feature_buffer.keys():
        if len(feature_buffer[person_id]) < SEQUENCE_LENGTH:
            if debug: print('Start Prediction Process')
            if debug: print("Duplicating the last feature to fill the buffer...")

            # Features
            num_duplicates_needed = SEQUENCE_LENGTH - len(feature_buffer[person_id])
            last_feature = feature_buffer[person_id][-1]
            duplicated_features = [last_feature] * num_duplicates_needed
            feature_buffer[person_id].extend(duplicated_features)

            # Frames 
            num_duplicates_needed = SEQUENCE_LENGTH - len(frames_buffer[person_id])
            last_frame = frames_buffer[person_id][-1]
            duplicated_frames = [last_frame] * num_duplicates_needed
            frames_buffer[person_id].extend(duplicated_frames)


            # ROIS
            num_duplicates_needed = SEQUENCE_LENGTH - len(rois_buffer[person_id])
            last_frame = rois_buffer[person_id][-1]
            duplicated_frames = [last_frame] * num_duplicates_needed
            rois_buffer[person_id].extend(duplicated_frames)


            if debug: print('Converting Buffers to Numpy')
            sequence_features = np.array(feature_buffer[person_id]).reshape(1, SEQUENCE_LENGTH, -1)
            sequence_frames = np.array([frames_buffer[person_id]])
            sequence_rois = np.array([rois_buffer[person_id]])

            # Aggregate predictions for voting
            final_predictions = model_predicts(lstm_model, dense_model, cnn_model, CLASSES_LIST, sequence_features, sequence_frames, sequence_rois, debug)
            if debug: print(f'Predicted: {final_predictions}')

            font_scale = 1
            (text_width, text_height), _ = cv2.getTextSize(final_predictions, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)

            # Define the starting position of the text: just above the bounding box or at the top of the frame if the bbox is too high.
            start_x = int((x1 + x2) / 2 - text_width / 2)
            start_y = y1 - 10  # 10 pixels above the bbox

            # Check if the text fits in the frame, if not adjust.
            if start_y - text_height < 0:  
                start_y = y1 + text_height + 10 
            
            # Define the coordinates for the rectangle
            rect_start_x = start_x
            rect_start_y = start_y - text_height
            rect_end_x = start_x + text_width
            rect_end_y = start_y

            # Draw the white rectangle
            # cv2.rectangle(frame, (rect_start_x, rect_start_y), (rect_end_x, rect_end_y), (255, 255, 255), -1)  # -1 means filled rectangle

            # Draw the text
            cv2.putText(frame, final_predictions, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)

            feature_buffer[person_id].pop(0)  # Slide the window
            frames_buffer[person_id].pop(0)
            rois_buffer[person_id].pop(0)

        frame_with_attention = overlay_heatmap_on_image(frame, frame_heatmap)
        combined_frame = np.hstack((frame_with_attention, frame))
        out.write(combined_frame)

    out.release()
    cap.release()