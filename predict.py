from frame_processing import detect_and_track
from movenet_helper import combined_heatmap
from fuse_features import fuse_features
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import tensorflow as tf


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.outputs = None
        self.model.eval()
        self.hook_layers()

    def hook_layers(self):
        def backward_hook_fn(module, grad_in, grad_out):
            self.gradients = grad_in[0]
            
        def forward_hook_fn(module, input, output):
            self.outputs = output

        self.target_layer.register_backward_hook(backward_hook_fn)
        self.target_layer.register_forward_hook(forward_hook_fn)

    def generate_cam(self, input_tensor, target_category=None):
        # Forward pass
        model_output = self.model(input_tensor)
        if target_category is None:
            target_category = torch.argmax(model_output, dim=1).item()
        
        # Zero gradients everywhere
        self.model.zero_grad()
        
        # Set the output for the target category to 1, and 0 for other categories
        one_hot_output = torch.zeros((1, model_output.shape[-1]))
        one_hot_output[0][target_category] = 1
        
        # Backward pass to get gradient information
        model_output.backward(gradient=one_hot_output)
        
        # Get the target layer's output after the forward pass
        target_layer_output = self.outputs[0]
        
        # Global Average Pooling (GAP) to get the weights
        weights = torch.mean(self.gradients, dim=(2, 3))[0, :]
        
        # Weighted combination to get the attention map
        cam = torch.zeros(target_layer_output.shape[1:]).to(input_tensor.device)
        for i, w in enumerate(weights):
            cam += w * target_layer_output[i, :, :]
        
        # ReLU to get only the positive values
        cam = nn.ReLU()(cam)
        
        # Resize the CAM to the input tensor's size
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        cam = torch.nn.functional.interpolate(cam.unsqueeze(0).unsqueeze(0), size=input_tensor.shape[2:], mode="bilinear").squeeze().cpu().detach().numpy()
        
        return cam

IMAGE_HEIGHT,IMAGE_WIDTH = 224,224

def extract_frames(video_file):
    capture = cv2.VideoCapture(video_file)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    EXTRACT_FREQUENCY = 2
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
                     grad_cam, 
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
    # video_dims = (frames[0].shape[1], frames[0].shape[0])
    video_dims = (frames[0].shape[1] * 2, frames[0].shape[0])
    output_video_path = output_video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, video_dims)

    def overlay_heatmap_on_image(image, heatmap, colormap=cv2.COLORMAP_JET, alpha=0.6):
        """
        Overlay a heatmap on an image.
        
        Parameters:
        - image: Original image.
        - heatmap: 2D numpy array representing the heatmap.
        - colormap: OpenCV colormap to apply to the heatmap.
        - alpha: The blending factor. 1.0 means only heatmap, 0.0 means only image.
        
        Returns:
        - Blended image.
        """
        # Resize heatmap to the size of the image
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        # Convert the heatmap to RGB
        heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), colormap)
        # Blend the image and the heatmap
        blended = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return blended
    
    def display_frame(frame):
        # Convert from BGR to RGB for proper display in matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)
        plt.axis('off')  # Hide axis values
        plt.show()

    
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
        # Process the ROI for ResNet and GradCAM
        if debug: print("Processing ROI...")
        resized_roi = cv2.resize(roi_frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_roi = resized_roi.astype(np.float32) / 255.0
        roi_tensor = torch.tensor(normalized_roi).permute(2, 0, 1).unsqueeze(0).float()

        # Generate attention heatmap for the ROI using GradCAM and ResNet
        roi_heatmap = grad_cam.generate_cam(roi_tensor)
        
        # Extract spatial and temporal features
        if debug: print("Extracting spatial and temporal features...")
        with torch.no_grad():
            spatial_features = spatial_resnet(roi_tensor).squeeze(-1).squeeze(-1)
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
            cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
            feature_buffer.pop(0)  # Slide the window
            frames_buffer.pop(0)
            
        # Overlay the heatmap on the frame
        frame_with_attention = overlay_heatmap_on_image(frame, roi_heatmap)
        # Concatenate the frame_with_attention (on the left) with the normal frame (on the right)
        combined_frame = np.hstack((frame_with_attention, frame))

        if debug:
            display_frame(combined_frame)

        out.write(combined_frame)
        
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
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)

        # Overlay the heatmap on the frame
        frame_with_attention = overlay_heatmap_on_image(frame, roi_heatmap)
        # Concatenate the frame_with_attention (on the left) with the normal frame (on the right)
        combined_frame = np.hstack((frame_with_attention, frame))

        if debug:
            display_frame(combined_frame)

        out.write(combined_frame)
    out.release()