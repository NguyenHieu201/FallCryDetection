import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Use this instead of from tensorflow import keras
# keras now is lazy loading module
keras = tf.keras

def preprocessing(config):
    tag = ['adl', 'fall']
    label_processor = keras.layers.StringLookup(
        num_oov_indices=0, vocabulary=pd.DataFrame(tag)[0]
    )
    activity_path = config["activity-path"]
    fall_path = config["fall-path"]

    feature_data = []

    for file in tqdm(os.listdir(activity_path)):
        file_path = os.path.join(activity_path, file)
        df = pd.read_csv(file_path)
        feature_data.append(df.to_numpy())
        
    for file in tqdm(os.listdir(fall_path)):
        file_path = os.path.join(fall_path, file)
        df = pd.read_csv(file_path)
        feature_data.append(df.to_numpy())
        
    feature_data = np.array(feature_data)
    label = ["adl"] * len(os.listdir(activity_path)) + ["fall"] * len(os.listdir(fall_path))
    return label_processor, feature_data, label


def split_data(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y, 
                                                        test_size=test_size)
    return X_train, X_test, y_train, y_test