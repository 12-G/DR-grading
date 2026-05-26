import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


def compute_dr_metrics(y_true, y_pred, num_classes: int = 5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    acc = accuracy_score(y_true, y_pred)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return {
        "acc": float(acc),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "qwk": float(qwk),
    }
