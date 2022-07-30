import json
import keras

from Utils.Preprocessing import preprocessing, split_data
from Utils.Metrics import metrics

from Models.LSTM import LSTM, classification_result
import numpy as np

from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot


config = json.load(open("setting.json", "r"))
epochs = config["hyper-parameters"]["epochs"]
batch_size = config["hyper-parameters"]["batch-size"]
test_size = config["test-size"]

save_path = config["save-path"]
checkpoint = keras.callbacks.ModelCheckpoint(
    save_path, save_weights_only=True, save_best_only=True, verbose=1
)

label_processor, feature_data, label = preprocessing(config)


label_numpy = label_processor(label).numpy()
X_train, X_test, y_train, y_test = split_data(X=feature_data, y=label_numpy, test_size=test_size)
feature_shape = X_train.shape

# Sampling
X = X_train.reshape(y_train.shape[0], -1)
y = y_train
over = SMOTE(sampling_strategy=0.75)
steps = [("o", over)]
pipeline = Pipeline(steps)
X, y = pipeline.fit_resample(X, y)
X = X.reshape(y.shape[0], feature_shape[1], feature_shape[2])

early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# Training phase
model = LSTM(config["hyper-parameters"])
history = model.fit(
    x=X,
    y=y,
    validation_split=0.3,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[checkpoint, early_stop]
)
# model.load_weights(save_path)
model.save("model.h5")
_, accuracy = model.evaluate(X_train, y_train)
print(f"Accuracy : {round(accuracy * 100, 2)}%")

# Evaluation metrics
prediction = classification_result(model, X_test)
true_value = y_test
for name, metric in metrics.items():
    print(f"\t{name}: {metric(prediction, true_value)}")