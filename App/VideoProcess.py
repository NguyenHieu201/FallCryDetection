import cv2
import keras
import numpy as np
import pandas as pd


label = ["adl", "fall"]
label_df = pd.DataFrame(label)
label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(label)
)

def load_video(path, max_frames, resize):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
            
            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)