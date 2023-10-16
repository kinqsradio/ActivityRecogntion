
# Activity Recognition

This repository contains the codebase for an advanced video analysis pipeline that combines multiple state-of-the-art models and methodologies for high-accuracy activity recognition. The pipeline utilizes models and methodologies such as YOLO V8, ByteTrack, Movenet, and Transformer encoders. It is trained on the Human Activity Recognition (HAR - Video Dataset).

## Project Pipeline

The pipeline encompasses the following steps:

1. **Frame Extraction**: Extract individual frames from video input to obtain spatial and temporal features.
2. **ROI Extraction and Tracking**: Use YOLO V8 for detecting the subject in the first frame and track the subject across subsequent frames.
3. **Pose Estimation**: Utilize Movenet for estimating the pose of the subject from the ROI images.
4. **Skeleton Heatmap Generation**: Convert the obtained skeleton poses to joint-limb skeleton heatmaps.
5. **Feature Engineering**: Extract spatial and temporal features from the ROI images and skeleton heatmaps using Transformer encoders.
6. **Feature Fusion**: Combine the spatial and temporal features for comprehensive representation.
7. **Activity Classification**: Use a classification model to predict the activity based on the fused features.

## Detailed Project Pipeline

1. **Frame Extraction**: Individual frames are extracted from video input to obtain spatial and temporal features.
   - Input: Video sequence
   - Action: Frame extraction using utilities
   - Output: Individual frames for subsequent processing

2. **ROI Extraction and Tracking**: The Region of Interest (ROI) is extracted, and subjects are tracked across frames.
   - Input: Extracted frames
   - Action: ROI extraction using YOLO V8 and subject tracking across frames
   - Output: Tracked subjects with corresponding ROIs

3. **Pose Estimation**: The pose of the subject within the ROIs is estimated.
   - Input: ROIs
   - Action: Pose estimation using Movenet
   - Output: Skeleton poses of the subject

4. **Skeleton Heatmap Generation**: The obtained skeleton poses are converted to a heatmap representation.
   - Input: Skeleton poses
   - Action: Conversion of poses to joint-limb skeleton heatmaps
   - Output: Skeleton heatmaps

5. **Feature Engineering**: Spatial and temporal features are extracted from the ROIs and skeleton heatmaps.
   - Input: ROIs, Skeleton heatmaps
   - Action: Feature extraction using Transformer encoders
   - Output: Spatial and temporal features

6. **Feature Fusion**: The spatial and temporal features are fused for a comprehensive representation.
   - Input: Spatial and temporal features
   - Action: Feature fusion
   - Output: Fused features

7. **Activity Classification**: The activity in the video sequence is predicted based on the fused features.
   - Input: Fused features
   - Action: Activity classification using the appropriate model
   - Output: Predicted activity class

## Detailed Function Descriptions

### create_datasets:
- Responsible for creating the datasets by processing videos, extracting frames, and generating features.
- Inputs: YOLO model, ResNet models, Transformer models, dataset directory, class names, sequence length, and number of videos to process.
- Operation: Frame extraction, feature generation, and data saving.
- Outputs: Processed video sequences, corresponding labels, and video file paths.

### create_cnn_datasets:
- Tailored for creating datasets suitable for CNN models.
- Specifics are similar to `create_datasets` but tailored for CNNs.

### predict_on_video:
- Predicts the class of a video and saves an annotated video with predicted actions.
- Inputs: Video path, output video path, class list, various models (dense, LSTM, CNN, YOLO, ResNet, Transformer), and sequence length.
- Operation: Frame processing, model inference, and video annotation.
- Outputs: An annotated video with predicted actions.

### detect_and_track:
- Integrates both detection and tracking capabilities to process video frames.
- Inputs: List of video frames and a model instance for detection and tracking.
- Operation: Object detection, tracking, ROI extraction, and skeletal pose highlighting.
- Outputs: Processed frame, extracted ROI, skeletal pose, and keypoints.


The function `create_datasets` in `datasets.py` can be summarized as follows:

### `create_datasets`:
- **Inputs**: 
  - `yolo_model`: A YOLO model instance used for object detection.
  - `spatial_resnet`: A ResNet model for spatial feature extraction.
  - `spatial_transformer`: Transformer model for spatial feature transformation.
  - `temporal_resnet`: ResNet model for temporal feature extraction.
  - `temporal_transformer`: Transformer model for temporal feature transformation.
  - `DATASET_DIR`: Directory path of the dataset.
  - `CLASSES_LIST`: List of class names.
  - `SEQUENCE_LENGTH`: Desired number of frames in each video sequence.
  - `NUM_VIDEOS_TO_PROCESS`: Number of videos to process.

- **Operation**:
  1. It first checks if the final features and labels already exist. If they do, it loads and returns them.
  2. If not, the function loads the previously processed data, if available.
  3. Processes each class in the `CLASSES_LIST`:
     - Gets the list of video files for the current class.
     - Processes each video:
       - Extracts frames.
       - Resizes frames and normalizes them.
       - Appends the resized frames to the features list and the class index to the labels list.
  4. Saves the processed data temporarily at regular intervals and at the end.
  
- **Outputs**:
  - `features`: Numpy array of processed video sequences.
  - `labels`: Numpy array of labels corresponding to each video sequence.
  - `video_files_paths`: List of video file paths corresponding to each video sequence.

The function `create_cnn_datasets` is not fully displayed due to space constraints. A brief overview based on the provided snippet is:

### `create_cnn_datasets`:
- Tailored for creating datasets suitable for CNN models.
- Specifics are similar to `create_datasets` but tailored for CNNs.

The function `predict_on_video` in `predict.py` can be summarized as follows:

### `predict_on_video`:
- **Inputs**: 
  - `video_path`: Path to the input video.
  - `output_video`: Path to save the output annotated video.
  - `CLASSES_LIST`: List of class names.
  - `dense_model`: Dense neural network model instance.
  - `lstm_model`: LSTM model instance.
  - `cnn_model`: CNN model instance.
  - `yolo_model`: A YOLO model instance used for object detection.
  - `spatial_resnet`: A ResNet model for spatial feature extraction.
  - `spatial_transformer`: Transformer model for spatial feature transformation.
  - `temporal_resnet`: ResNet model for temporal feature extraction.
  - `temporal_transformer`: Transformer model for temporal feature transformation.
  - `SEQUENCE_LENGTH`: Desired number of frames in each video sequence.
  - `debug`: A boolean flag for debugging purposes.

- **Operation**: 
  1. **Initialization**: Prepare buffers to hold extracted features and frames.
  2. **Combined Voting**: Define a strategy to combine predictions from different models through both soft and hard voting.
  3. **Frame Extraction and Preprocessing**: Extract frames from the video, detect and track subjects, generate keypoint heatmaps, and preprocess the Region of Interest (ROI).
  4. **Feature Extraction and Transformation**: Derive spatial and temporal features from the ROI and heatmap, then transform these features using Transformer models.
  5. **Feature Fusion and Prediction**: Combine spatial and temporal features. When enough features are collected (equal to `SEQUENCE_LENGTH`), predictions are made using LSTM, Dense, and CNN models. These predictions are combined using the voting strategy.
  6. **Annotation**: Annotate the video frame with the predicted action.
  7. **End-of-video Handling**: For remaining frames at the end of the video, duplicate the last feature and frame to predict on them.

- **Outputs**: 
  - An annotated video saved at the path specified by `output_video`, where each frame is annotated with the predicted action from the provided models.

The function `detect_and_track` in `frame_processing.py` can be summarized as follows:

### `detect_and_track`:
- **Inputs**: 
  - `frames`: A list of video frames.
  - `model`: A model instance used for object detection and tracking.

- **Operation**:
  1. Retrieves class names from the model.
  2. For each frame in `frames`:
     - Runs detection and tracking using the provided model.
     - Retrieves the class indices, bounding boxes, and ids of detected objects.
     - Filters the detections to only consider 'person' objects.
     - Selects the best bounding box (possibly using some criteria, details not shown).
     - Extracts the Region of Interest (ROI) based on the best bounding box.
     - Runs inference on the ROI to obtain keypoints.
     - Draws predictions on the frame based on the keypoints.
     - Resizes the frame, ROI, skeleton pose, and poses on the frame.
     - Yields the processed frame, ROI, skeleton pose, and poses on the frame, as well as keypoints for further processing.
  
- **Outputs**:
  - A generator that yields:
    - `frame`: The processed video frame.
    - `roi`: The extracted Region of Interest.
    - `skeleton_pose`: An image depicting the skeletal pose.
    - `poses_on_frame`: The frame annotated with the skeletal pose.
    - `keypoints_with_scores`: The keypoints detected in the frame.

## Dataset

The model is trained on the [Human Activity Recognition (HAR - Video Dataset)](https://www.kaggle.com/datasets/sharjeelmazhar/human-activity-recognition-video-dataset?resource=download-directory) from Kaggle.