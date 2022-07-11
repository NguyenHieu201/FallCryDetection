import json
import numpy as np
import keras
from Models.LSTM import LSTM, classification_result
from Utils.Preprocessing import preprocessing, split_data
from Utils.Metrics import metrics


config = json.load(open("setting.json", "r"))
data_path = config["data-path"]
epochs = config["hyper-parameters"]["epochs"]
batch_size = config["hyper-parameters"]["batch-size"]
test_size = config["test-size"]


filepath = "Results/"
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath, save_weights_only=True, save_best_only=True, verbose=1
)

label_processor, feature_data, label = preprocessing(data_path)
X_train, X_test, y_train, y_test = split_data(X=feature_data, y=label, test_size=test_size)

# Training phase
model = LSTM(config["hyper-parameters"])
history = model.fit(
    x=X_train,
    y=y_train,
    validation_split=0.3,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[checkpoint]
)
model.load_weights(filepath)
_, accuracy = model.evaluate(feature_data, label)
print(f"Accuracy : {round(accuracy * 100, 2)}%")

# Evaluation metrics
prediction = classification_result(model, X_test)
true_value = y_test
for name, metric in metrics.items():
    print(f"\t{name}: {metric(prediction, true_value)}")