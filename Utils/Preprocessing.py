import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Use this instead of from tensorflow import keras
# keras now is lazy loading module
keras = tf.keras

def preprocessing(data_path):
    label_df = pd.read_csv(os.path.join(data_path, "label.csv"), index_col=0)
    label = label_df.to_numpy()
    n_sample = label.shape[0]
    tag = ['adl', 'fall']
    label_processor = keras.layers.StringLookup(
        num_oov_indices=0, vocabulary=pd.DataFrame(tag)[0]
    )
    
    # Get input for model
    feature_data = []
    mask_data = []
    for i in tqdm(range(n_sample)):
        feature_path = os.path.join(data_path, "Feature", f"{i}.csv")
        mask_path = os.path.join(data_path, "Mask", f"{i}.csv")
        feature_df = pd.read_csv(feature_path)
        mask_df = pd.read_csv(mask_path)
        feature_data.append(feature_df.to_numpy())
        mask_data.append(mask_df.to_numpy())

    feature_data = np.array(feature_data)
    mask_data = np.array(mask_data)
    return label_processor, feature_data, mask_data, label


def split_data(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y, 
                                                        test_size=test_size)
    return X_train, X_test, y_train, y_test