import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    jaccard_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
    zero_one_loss,
)

# ----------------------------------------------------------------------------------------------------
# Multi-Label Metrics


# def jaccard_index(true, pred):
#     inter = (true & pred).float().sum(dim=1)
#     union = (true | pred).float().sum(dim=1)
#     return (inter / union).mean().item()


# def hamming_loss(true, pred):
#     return (pred != true).float().mean().item()


def macro_accuracy(true, pred):
    true = np.asarray(true, dtype=bool)
    pred = np.asarray(pred, dtype=bool)

    # (pred == true).astype(float).mean().item()

    tp = (pred & true).sum(axis=0)
    tn = ((1 - pred) & (1 - true)).sum(axis=0)
    fp = (pred & (1 - true)).sum(axis=0)
    fn = ((1 - pred) & true).sum(axis=0)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return accuracy.mean().item()


def micro_accuracy(true, pred):
    true = np.asarray(true, dtype=bool)
    pred = np.asarray(pred, dtype=bool)

    # (pred == true).astype(float).mean().item()

    tp = (pred & true).sum()
    tn = ((1 - pred) & (1 - true)).sum()
    fp = (pred & (1 - true)).sum()
    fn = ((1 - pred) & true).sum()
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return accuracy.item()


def per_label_accuracy(true, pred):
    true = np.asarray(true)
    pred = np.asarray(pred)

    return (pred == true).astype(float).mean(axis=0).tolist()


# def subset_accuracy(true, pred):
#     return (pred == true).all(dim=1).float().mean().item()


def compute_multihot_metrics(y_true, y_pred, num_classes=None):

    a_subset = accuracy_score(y_true, y_pred)
    a_micro = micro_accuracy(y_true, y_pred)
    a_macro = macro_accuracy(y_true, y_pred)
    a_per_class = per_label_accuracy(y_true, y_pred)

    p_micro_ = precision_score(y_true, y_pred, average="micro", zero_division=np.nan)
    p_macro_ = precision_score(y_true, y_pred, average="macro", zero_division=np.nan)
    p_samples_ = precision_score(
        y_true, y_pred, average="samples", zero_division=np.nan
    )
    p_per_class_ = precision_score(y_true, y_pred, average=None, zero_division=np.nan)

    r_micro_ = recall_score(y_true, y_pred, average="micro", zero_division=np.nan)
    r_macro_ = recall_score(y_true, y_pred, average="macro", zero_division=np.nan)
    r_samples_ = recall_score(y_true, y_pred, average="samples", zero_division=np.nan)
    r_per_class_ = recall_score(y_true, y_pred, average=None, zero_division=np.nan)

    f1_micro_ = f1_score(y_true, y_pred, average="micro", zero_division=np.nan)
    f1_macro_ = f1_score(y_true, y_pred, average="macro", zero_division=np.nan)
    f1_samples_ = f1_score(y_true, y_pred, average="samples", zero_division=np.nan)
    f1_per_class_ = f1_score(y_true, y_pred, average=None, zero_division=np.nan)

    jacc = jaccard_score(y_true, y_pred, average="samples")

    hamm_loss = hamming_loss(y_true, y_pred)
    zero_one_loss_ = zero_one_loss(y_true, y_pred)

    # dim = num_classes

    return {
        "accuracy": a_subset,
        "accuracy_micro": a_micro,
        "accuracy_macro": a_macro,
        "accuracy_per_class": a_per_class,
        "precision_micro": p_micro_,
        "precision_macro": p_macro_,
        "precision_samples": p_samples_,
        "precision_per_class": p_per_class_,
        "recall_micro": r_micro_,
        "recall_macro": r_macro_,
        "recall_samples": r_samples_,
        "recall_per_class": r_per_class_,
        "f1_micro": f1_micro_,
        "f1_macro": f1_macro_,
        "f1_samples": f1_samples_,
        "f1_per_class": f1_per_class_,
        "jaccard_score": jacc,
        "hamming_loss": hamm_loss,
        "zero_one_loss": zero_one_loss_,
    }


# ----------------------------------------------------------------------------------------------------
# Multi-Class Metrics


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
