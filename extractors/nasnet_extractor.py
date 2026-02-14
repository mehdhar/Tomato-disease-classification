import numpy as np
from tensorflow.keras.applications import NASNetMobile

def extract_nasnet_features(images):
    model = NASNetMobile(weights="imagenet", include_top=False)
    features = model.predict(images, verbose=1)
    return features.reshape(features.shape[0], -1)
