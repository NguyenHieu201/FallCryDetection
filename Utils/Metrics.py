import keras
metrics = {
    "True Positive": keras.metrics.TruePositives(name='tp'),
    "False Positives": keras.metrics.FalsePositives(name='fp'),
    "True Negatives": keras.metrics.TrueNegatives(name='tn'),
    "False Negatives": keras.metrics.FalseNegatives(name='fn'), 
    "Binary Accuracy": keras.metrics.BinaryAccuracy(name='accuracy'),
    "Precision": keras.metrics.Precision(name='precision'),
    "Recall": keras.metrics.Recall(name='recall'),
    "AUC": keras.metrics.AUC(name='auc'),
    "Precision-Recall curve": keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
}