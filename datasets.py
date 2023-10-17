# Standard library imports
import os
import gc
import logging

# Third-party library imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from keras.preprocessing.sequence import pad_sequences
from sklearn.decomposition import PCA

# Local imports
from fuse_features import fuse_features
from frame_processing import detect_and_track
from movenet_helper import combined_heatmap
from IPython.display import clear_output


IMAGE_HEIGHT,IMAGE_WIDTH = 224,224

def save_temp_data(features, labels, video_files_paths, prefix="temp"):
    """Utility function to save data temporarily."""
    np.save(f'{prefix}_features.npy', features)
    np.save(f'{prefix}_labels.npy', labels)
    with open(f'{prefix}_video_files_paths.txt', 'w') as f:
        for path in video_files_paths:
            f.write(f"{path}\n")
    
def find_last_saved_class_and_video(NUM_VIDEOS_TO_PROCESS, CLASSES_LIST):
    """Find the class and index of the last saved video."""
    for class_index in reversed(range(len(CLASSES_LIST))):
        class_name = CLASSES_LIST[class_index]
        index = NUM_VIDEOS_TO_PROCESS-1  # since we process only first 10 videos
        while index >= 0:
            path = f'temp_{class_name}_{index}_features.npy'
            file_exists = bool(os.path.exists(path))
            print(f'Checking last saved exist temp_{class_name}_{index}_features.npy: {file_exists}')
            if os.path.exists(f'temp_{class_name}_{index}_features.npy'):
                return class_name, index  # Return the class and the last saved video index
            index -= 1
    return None, None  # If no saved data is found


def extract_frames(video_path, SEQUENCE_LENGTH):
    frames_list = []

    # Open the video file and get the total frame count
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine how many frames to skip to achieve desired sequence length
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)

    for frame_counter in range(SEQUENCE_LENGTH):
        # Set the next frame to read based on skip window
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()

        # Stop if no more frames are available
        if not success:
            break
        
        # Append the normalized frame to the list
        frames_list.append(frame)
    
    video_reader.release()

    gc.collect()

    return frames_list

def create_datasets(yolo_model, 
                    spatial_resnet, 
                    spatial_transformer, 
                    temporal_resnet, 
                    temporal_transformer,
                    DATASET_DIR,
                    CLASSES_LIST,
                    SEQUENCE_LENGTH,
                    NUM_VIDEOS_TO_PROCESS):
    
    # Check if final data exists
    if os.path.exists('final_features.npy') and os.path.exists('final_labels.npy') and os.path.exists('final_video_files_paths.txt'):
        print("Final data already exists. Loading and skipping further processing...")
        features = np.load('final_features.npy', allow_pickle=True)
        labels = np.load('final_labels.npy', allow_pickle=True)
        with open('final_video_files_paths.txt', 'r') as f:
            video_files_paths = [line.strip() for line in f.readlines()]
        
        print(f'Features shape: {features.shape}')
        print(f'Labels shape: {labels.shape}\n')

        # Padding to the correct output
        # The length to which all sequences will be padded or truncated
        MAX_SEQUENCE_LENGTH = max([f.shape[0] for f in features])

        # Pad sequences
        features_padded = pad_sequences(features, maxlen=MAX_SEQUENCE_LENGTH, padding='post', dtype='float32')

        print(f'Features shape: {features_padded.shape}') # Expected (number_of_videos, MAX_SEQUENCE_LENGTH, feature_dimension)
        print(f'Labels shape: {labels.shape}') # Expected(number_of_videos)

        return np.array(features_padded), np.array(labels), video_files_paths
    
    # Step 1: Load the Previous Data
    last_class, last_video = find_last_saved_class_and_video(NUM_VIDEOS_TO_PROCESS, CLASSES_LIST)
    if last_class and last_video is not None:
        features = list(np.load(f'temp_{last_class}_{last_video}_features.npy', allow_pickle=True))
        print(f'Loading: temp_{last_class}_{last_video}_features.npy')
        print(len(features))
        labels = list(np.load(f'temp_{last_class}_{last_video}_labels.npy', allow_pickle=True))
        print(f'Loading: temp_{last_class}_{last_video}_labels.npy')
        print(len(labels))
        with open(f'temp_{last_class}_{last_video}_video_files_paths.txt', 'r') as f:
            print(f'Loading: temp_{last_class}_{last_video}_video_files_paths.txt')
            video_files_paths = [line.strip() for line in f.readlines()]
    else:
        features = []
        labels = []
        video_files_paths = []

    SAVE_INTERVAL = 1  # Save every 1
    
    # Step 2: Determine the Starting Point for Processing
    start_class = 0 if not last_class else CLASSES_LIST.index(last_class)
    start_video = 0 if not last_video else last_video + 1

    for class_index in range(start_class, len(CLASSES_LIST)):
        class_name = CLASSES_LIST[class_index]
        gc.collect()
        files_list = os.listdir(os.path.join(DATASET_DIR, class_name))
        
        # Determine starting point based on last saved video
        start_video = start_video if class_name == last_class else 0
        
        files_list = files_list[start_video:min(start_video+NUM_VIDEOS_TO_PROCESS-len(files_list), len(files_list))]
        print(f'start_video+NUM_VIDEOS_TO_PROCESS-len(files_list): {start_video+NUM_VIDEOS_TO_PROCESS-len(files_list)}')
        print(f'Length files_list: {len(files_list)}')

        # Step 3: Continue Processing
        for video_count, file_name in enumerate(files_list):
            video_count += start_video
            print(f'Video Count: {video_count}')
            if video_count == NUM_VIDEOS_TO_PROCESS:
                print('Moving to next Class')
                break
            
            print(f'Starting extraction for Class: {class_name} Video {video_count}')
            video_file_path = os.path.join(DATASET_DIR, class_name, file_name)
            frames = extract_frames(video_file_path, SEQUENCE_LENGTH)
            print(f'Number of frames: {len(frames)}')

            # Unpacked processed data
            processed_data = list(detect_and_track(frames, yolo_model))

            try:
                frames_list, roi_frames, skeleton_poses, _, keypoints_with_scores = zip(*processed_data)
                print('Processed data unpacked.')
            except ValueError:
                print(f"Error processing video {video_file_path}. Skipping to next video...")
                continue

            frames_list = list(frames_list)
            roi_frames = list(roi_frames)
            skeleton_poses = list(skeleton_poses)

            # Generate heatmaps directly from keypoints_with_scores
            heatmaps_list = [combined_heatmap(pose[0, 0], IMAGE_HEIGHT, IMAGE_WIDTH) for pose in keypoints_with_scores]

            # Convert heatmaps to tensors and normalize
            heatmaps = torch.stack([F.interpolate(torch.tensor(heatmap).unsqueeze(0).unsqueeze(0),
                                                size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode='bilinear').squeeze()
                                    for heatmap in heatmaps_list]).unsqueeze(1).numpy()

            # Safe normalization
            max_value = np.max(heatmaps)
            if max_value > 0:  # Avoid division by zero
                heatmaps = heatmaps / max_value

            heatmaps_tensor = torch.tensor(heatmaps).float().repeat(1, 3, 1, 1)
            print('Heatmaps generated.')

            # ROIS
            roi_imgs = torch.stack([F.interpolate(torch.tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).unsqueeze(0).unsqueeze(0),
                                            size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode='bilinear').squeeze()
                                for frame in roi_frames]).unsqueeze(1).numpy()
            # Normalizing roi_imgs
            roi_imgs = roi_imgs / 255.0
            roi_imgs_tensor = torch.tensor(roi_imgs).float().repeat(1, 3, 1, 1)
            
            print('ROIs generated.')

            # Display some sample ROIs
            sample_count = 5
            for idx, (frame, heatmap) in enumerate(zip(roi_frames, heatmaps)):
                if idx >= sample_count:
                    break
                
                print('Displaying Samples')
                
                # Display the heatmap
                plt.imshow(heatmap[0], cmap='hot', interpolation='nearest')
                plt.title("Heatmap")
                plt.show()

                # Display the ROI
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cmap='gray')
                plt.title("ROI")
                plt.show()


            # Features extraction using Resnet (Spatial) (Temporal)
            print('Features extracted using ResNet.')
            with torch.no_grad():
                spatial_features = spatial_resnet(roi_imgs_tensor).squeeze(-1).squeeze(-1)
                temporal_features = temporal_resnet(heatmaps_tensor).squeeze(-1).squeeze(-1)

            # Attentions
            print('Features processed through Transformer Encoders.')
            print('Spatial Features Transformers')
            spatial_features = spatial_transformer(spatial_features)
            print(spatial_features)
            
            print('Temporal Features Transformers')
            temporal_features = temporal_transformer(temporal_features)
            print(temporal_features)

            # Combined Features
            print('Combined Features')
            combined_features = fuse_features(spatial_features.detach().numpy(), temporal_features.detach().numpy())

            # Normalized Combined Features
            print('Normalized Combined Features')
            mean = np.mean(combined_features, axis=0)
            std = np.std(combined_features, axis=0)
            combined_features = (combined_features - mean) / std
            
            print(combined_features)

            # Appending to training list
            clear_output(wait=True)
            features.append(combined_features)
            labels.append(class_index)
            video_files_paths.append(video_file_path)
            
            print("Shape of spatial features:", spatial_features.shape)
            print("Shape of temporal features:", temporal_features.shape)
            print("Shape of combined_faeatures:", combined_features.shape)
            print(f'Appended features for Class: {class_name} Video {video_count}')

            # Check if it's time to save
            if video_count % SAVE_INTERVAL == 0:
                print(f"Attempting to save data for Class: {class_name} Video {video_count}")
                save_temp_data(features, labels, video_files_paths, prefix=f"temp_{class_name}_{video_count}")
                print(f"Data saved for Class: {class_name} Video {video_count}")

            del frames, frames_list, roi_frames, skeleton_poses, keypoints_with_scores
            gc.collect()

    # Save one last time at the end
    save_temp_data(features, labels, video_files_paths, prefix="final")
    
    # Padding to the correct output
    # The length to which all sequences will be padded or truncated
    MAX_SEQUENCE_LENGTH = max([f.shape[0] for f in features])

    # Pad sequences
    features_padded = pad_sequences(features, maxlen=MAX_SEQUENCE_LENGTH, padding='post', dtype='float32')

    print(f'Features shape: {features_padded.shape}') # Expected (number_of_videos, MAX_SEQUENCE_LENGTH, feature_dimension)

    return np.array(features_padded), np.array(labels), video_files_paths


def create_cnn_datasets(DATASET_DIR,
                    CLASSES_LIST,
                    SEQUENCE_LENGTH,
                    NUM_VIDEOS_TO_PROCESS):
    
    # Check if final data exists
    if os.path.exists('final_features.npy') and os.path.exists('final_labels.npy') and os.path.exists('final_video_files_paths.txt'):
        print("Final data already exists. Loading and skipping further processing...")
        features = np.load('final_features.npy', allow_pickle=True)
        labels = np.load('final_labels.npy', allow_pickle=True)
        with open('final_video_files_paths.txt', 'r') as f:
            video_files_paths = [line.strip() for line in f.readlines()]
        
        print(f'Features shape: {features.shape}')
        print(f'Labels shape: {labels.shape}\n')

        return np.array(features), np.array(labels), video_files_paths
    
    # Step 1: Load the Previous Data
    last_class, last_video = find_last_saved_class_and_video(NUM_VIDEOS_TO_PROCESS, CLASSES_LIST)
    if last_class and last_video is not None:
        features = list(np.load(f'temp_{last_class}_{last_video}_features.npy', allow_pickle=True))
        print(f'Loading: temp_{last_class}_{last_video}_features.npy')
        print(len(features))
        labels = list(np.load(f'temp_{last_class}_{last_video}_labels.npy', allow_pickle=True))
        print(f'Loading: temp_{last_class}_{last_video}_labels.npy')
        print(len(labels))
        with open(f'temp_{last_class}_{last_video}_video_files_paths.txt', 'r') as f:
            print(f'Loading: temp_{last_class}_{last_video}_video_files_paths.txt')
            video_files_paths = [line.strip() for line in f.readlines()]
    else:
        features = []
        labels = []
        video_files_paths = []

    SAVE_INTERVAL = 1  # Save every 1
    
    # Step 2: Determine the Starting Point for Processing
    start_class = 0 if not last_class else CLASSES_LIST.index(last_class)
    start_video = 0 if not last_video else last_video + 1

    for class_index in range(start_class, len(CLASSES_LIST)):
        class_name = CLASSES_LIST[class_index]
        gc.collect()
        files_list = os.listdir(os.path.join(DATASET_DIR, class_name))
        
        # Determine starting point based on last saved video
        start_video = start_video if class_name == last_class else 0
        
        files_list = files_list[start_video:min(start_video+NUM_VIDEOS_TO_PROCESS-len(files_list), len(files_list))]
        print(f'start_video+NUM_VIDEOS_TO_PROCESS-len(files_list): {start_video+NUM_VIDEOS_TO_PROCESS-len(files_list)}')
        print(f'Length files_list: {len(files_list)}')

        # Step 3: Continue Processing
        for video_count, file_name in enumerate(files_list):
            video_count += start_video
            print(f'Video Count: {video_count}')
            if video_count == NUM_VIDEOS_TO_PROCESS:
                print('Moving to next Class')
                break
            
            print(f'Starting extraction for Class: {class_name} Video {video_count}')
            video_file_path = os.path.join(DATASET_DIR, class_name, file_name)
            frames = extract_frames(video_file_path, SEQUENCE_LENGTH)
            print(f'Number of frames: {len(frames)}')

            if len(frames) == SEQUENCE_LENGTH:
                resized_frames = [(cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT)).astype(np.float32) / 255.0) for frame in frames]  # Normalize to [0, 1]
                features.append(resized_frames)
                labels.append(class_index)
                video_files_paths.append(video_file_path)

            # Check if it's time to save
            if video_count % SAVE_INTERVAL == 0:
                print(f"Attempting to save data for Class: {class_name} Video {video_count}")
                save_temp_data(features, labels, video_files_paths, prefix=f"temp_{class_name}_{video_count}")
                print(f"Data saved for Class: {class_name} Video {video_count}")

            del frames
            gc.collect()

    # Save one last time at the end
    save_temp_data(features, labels, video_files_paths, prefix="final")

    return np.array(features), np.array(labels), video_files_paths

# Data Augment
def inject_noise(data, noise_factor=0.05):
    noise = np.random.randn(*data.shape) * noise_factor
    augmented_data = data + noise
    return np.clip(augmented_data, 0, 1)  # Assuming data is normalized between 0 and 1

def time_warp(data, warp_factor=0.1):
    time_steps = data.shape[1]
    warp_size = int(time_steps * warp_factor)
    if warp_size == 0:
        return data
    
    if np.random.rand() < 0.5:
        return np.concatenate((data[:, :warp_size], data), axis=1)[:, :-warp_size]
    else:
        return np.concatenate((data, data[:, -warp_size:]), axis=1)[:, warp_size:]

def feature_jitter(data, jitter_factor=0.05):
    jitter = (np.random.rand(*data.shape) * 2 - 1) * jitter_factor
    return data + jitter


def random_crop(data, crop_factor=0.9):
    """Randomly crop the sequence."""
    seq_len = data.shape[1]
    new_len = int(seq_len * crop_factor)
    start_index = np.random.randint(0, seq_len - new_len)
    return data[:, start_index:start_index+new_len, :]

def time_masking(data, mask_factor=0.1):
    """Mask some time steps of the sequence."""
    seq_len = data.shape[1]
    mask_len = int(seq_len * mask_factor)
    mask_start = np.random.randint(0, seq_len - mask_len)
    data[:, mask_start:mask_start+mask_len, :] = 0
    return data

def temporal_scaling(data, scale_factor):
    new_length = int(data.shape[1] * scale_factor)
    scaled_data = np.zeros((data.shape[0], new_length, data.shape[2]))
    for i in range(data.shape[0]):
        scaled_data[i] = cv2.resize(data[i], (data.shape[2], new_length), interpolation=cv2.INTER_LINEAR)
    return scaled_data

def pad_sequences_to_length(data, target_length):
    current_length = data.shape[1]
    if current_length < target_length:
        padding = np.zeros((data.shape[0], target_length - current_length, data.shape[2]))
        return np.concatenate([data, padding], axis=1)
    return data

def augment_features(features):
    original_length = features.shape[1]
    all_augmented = []
    
    # Inject noise
    all_augmented.append(inject_noise(features))

    # Time warp
    all_augmented.append(time_warp(features))

    # Feature jitter
    all_augmented.append(feature_jitter(features))

    # Random crop
    cropped_features = random_crop(features)
    all_augmented.append(pad_sequences_to_length(cropped_features, original_length))

    # Time masking
    all_augmented.append(time_masking(features))

    # Temporal scaling
    scale_factor = 0.9  # For demonstration; you might want to randomly vary this for each sequence
    scaled_features = temporal_scaling(features, scale_factor)
    all_augmented.append(pad_sequences_to_length(scaled_features, original_length))
    
    return np.concatenate(all_augmented, axis=0)

# For Visualising
def visualize(original, augmented, title="Augmentation"):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(original[0, :, 0])
    plt.title("Original Sequence")
    
    plt.subplot(1, 2, 2)
    plt.plot(augmented[0, :, 0])
    plt.title(title)
    
    plt.tight_layout()
    plt.show()
    
def visualize_inject_noise(features):
    augmented_features = inject_noise(features)
    visualize(features, augmented_features, "Noisy Sequence")

def visualize_time_warp(features):
    augmented_features = time_warp(features)
    visualize(features, augmented_features, "Time Warped Sequence")

def visualize_feature_jitter(features):
    augmented_features = feature_jitter(features)
    visualize(features, augmented_features, "Jittered Sequence")

def visualize_random_crop(features):
    augmented_features = random_crop(features)
    augmented_features = pad_sequences_to_length(augmented_features, features.shape[1])
    visualize(features, augmented_features, "Cropped Sequence")

def visualize_time_masking(features):
    augmented_features = time_masking(features)
    visualize(features, augmented_features, "Time Masked Sequence")

def visualize_temporal_scaling(features):
    augmented_features = temporal_scaling(features, 0.9)
    augmented_features = pad_sequences_to_length(augmented_features, features.shape[1])
    visualize(features, augmented_features, "Temporally Scaled Sequence")

def visualize_mixed_augmentation(features):
    augmented_features = augment_features(features)
    visualize(features, augmented_features[0].reshape(1, features.shape[1], features.shape[2]), "Mixed Augmented Sequence")