import numpy as np

def categorical_accuracy(Y_true, Y_pred) -> float:
    """
    Computes the accuracy of predictions for a categorical classification problem.

    Args:
        Y_true (scalar or np.array): The true class labels, a scalar or an array of size (num_samples,). Each value must range 
            from 0 to num_classes-1.
        Y_pred (np.array): The predicted probabilities of each class. Array of size (num_classes,) or (num_samples, num_classes).
    
    Returns:
        score (float): The accuracy score, the fraction of correct predictions from the total number of predictions.
    """
    if len(Y_pred.shape) == 1:
        return 1. if np.argmax(Y_pred) == Y_true else 0.
    else:
        Y_pred_class = np.argmax(Y_pred, axis=1)
        return np.sum(Y_pred_class == Y_true) / len(Y_true)
