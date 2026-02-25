import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)


def compute_multiclass_metrics(y_true, y_pred, num_classes=None, class_weights=None):
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    # Accuracy per class
    tp = cm.diagonal()  # True Positives for each class
    fn = cm.sum(axis=1) - tp  # False Negatives for each class
    fp = cm.sum(axis=0) - tp  # False Positives for each class
    tn = cm.sum() - (tp + fn + fp)  # True Negatives for each class
    accuracy_per_class = (tp + tn) / (tp + tn + fp + fn)  # shape: (num_classes,)

    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    precision_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
    precision_weighted = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
        sample_weight=class_weights,
    )
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)

    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average="micro", zero_division=0)
    recall_weighted = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
        sample_weight=class_weights,
    )
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)

    return {
        "accuracy": accuracy,
        "accuracy_per_class": accuracy_per_class,
        "precision_macro": precision_macro,
        "precision_micro": precision_micro,
        "precision_weighted": precision_weighted,
        "precision_per_class": precision_per_class,
        "recall_macro": recall_macro,
        "recall_micro": recall_micro,
        "recall_weighted": recall_weighted,
        "recall_per_class": recall_per_class,
        "confusion_matrix": cm,
    }


def df_metrics_per_class(metrics, class_names=None):
    return pd.DataFrame(
        {
            "Accuracy": metrics["accuracy_per_class"],
            "Precision": metrics["precision_per_class"],
            "Recall": metrics["recall_per_class"],
        },
        index=class_names,
    ).round(3)
