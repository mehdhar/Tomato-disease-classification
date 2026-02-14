from .mobilenet_extractor import extract_mobilenet_features
from .nasnet_extractor import extract_nasnet_features
import numpy as np

def extract_hybrid_features(images):
    mobilenet_f = extract_mobilenet_features(images)
    nasnet_f = extract_nasnet_features(images)
    return np.concatenate([mobilenet_f, nasnet_f], axis=1)
