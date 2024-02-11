import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    
    precision_score = true_positive / (true_positive + false_positive + 1e-9)
    recall_score = true_positive / (true_positive + false_negative + 1e-9)
    f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score + 1e-9)
    accuracy_score = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    return precision_score, recall_score, f1_score, accuracy_score


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    """
    YOUR CODE IS HERE
    """
    unique_classes = np.unique(np.concatenate((y_true, y_pred)))
    num_classes = len(unique_classes)
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_true)):
        true_class_idx = np.where(unique_classes == y_true[i])[0][0]
        pred_class_idx = np.where(unique_classes == y_pred[i])[0][0]
        confusion_matrix[true_class_idx, pred_class_idx] += 1
    
    precision_score = np.sum(np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=0) + 1e-9)) / num_classes
    recall_score = np.sum(np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + 1e-9)) / num_classes
    f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score + 1e-9)
    accuracy_score = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    return precision_score, recall_score, f1_score, accuracy_score


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    y_mean = sum(y_true) / len(y_true)
    ssr = sum((y_pred - y_true) ** 2)
    sst = sum((y_true - y_mean) ** 2)
    r2 = 1 - (ssr / sst)
    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    mse = sum((y_pred - y_true) ** 2) / len(y_true)
    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    mae = sum(abs(y_pred - y_true)) / len(y_true)
    return mae
    