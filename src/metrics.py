import numpy as np


def accuracy(y_true, y_pred):
    """Compute classification accuracy without sklearn.metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred):
    """Return [[TN, false alarms], [missed faults, TP]] for labels 0 and 1."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_alarms = int(np.sum((y_true == 0) & (y_pred == 1)))
    missed_faults = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, false_alarms], [missed_faults, tp]], dtype=int)
