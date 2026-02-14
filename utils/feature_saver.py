import numpy as np

def save_features(path, features, labels=None):
    np.save(path + "_features.npy", features)
    if labels is not None:
        np.save(path + "_labels.npy", labels)

def load_features(path):
    features = np.load(path + "_features.npy")
    try:
        labels = np.load(path + "_labels.npy")
        return features, labels
    except:
        return features
