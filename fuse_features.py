import numpy as np

def fuse_features(spatial_features, temporal_features):
    fused_features = np.concatenate([spatial_features, temporal_features], axis=-1)
    return fused_features