import keras
import numpy as np

def LSTM(config):
    seq_len = config["seq-len"]
    num_feature = config["num-feature"]
    feature_input = keras.Input((seq_len, num_feature))
    
    x = keras.layers.GRU(16, return_sequences=True)(feature_input)
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(2, activation="softmax")(x)
    
    model = keras.Model(feature_input, output)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics = ["accuracy"]
            
    )
    return model

def classification_result(model, data):
    model_pred = model.predict(data)
    prediction = np.argmax(model_pred, axis=1)
    return prediction
    