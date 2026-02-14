import os, glob
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from .paths import TRAIN_DIR, VALID_DIR, SIZE

def load_dataset(train_dir=TRAIN_DIR, valid_dir=VALID_DIR):
    
    def load_split(path):
        images, labels = [], []
        for class_dir in glob.glob(path + "/*"):
            label = os.path.basename(class_dir)
            for img_path in glob.glob(class_dir + "/*.jpg"):
                img = cv2.imread(img_path)
                img = cv2.resize(img, (SIZE, SIZE))
                img = img / 255.0
                images.append(img)
                labels.append(label)
        return np.array(images), np.array(labels)

    x_train, y_train = load_split(train_dir)
    x_test, y_test = load_split(valid_dir)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    return x_train, y_train, x_test, y_test, encoder
