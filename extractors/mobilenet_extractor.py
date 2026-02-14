import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

def extract_mobilenet_features(images):
    model = MobileNetV2(weights="imagenet", include_top=False)
    features = model.predict(images, verbose=1)
    return features.reshape(features.shape[0], -1)
