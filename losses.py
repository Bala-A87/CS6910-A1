import numpy as np

class LossFunction():
    """
    Template class to implement a loss function. Each chuld class must implement two methods, `forward` and `backward`.

    `forward` accepts the predicted probabilities (np.array of shape (num_classes,) or (num_samples, num_classes)) and the 
    true class labels (scalar or np.array of shape (num_samples,)) and returns the loss (or average loss).

    `backward` accepts the predicted probabilities (np.array of shape (num_classes,) or (num_samples, num_classes)) and the 
    true class labels (scalar or np.array of shape (num_samples,)) and returns the gradient of the loss wrt each of the outputs,
    an np.array of shape matching the shape of the predictions.
    """
    def __init__(self) -> None:
        pass

    def forward(self, y_pred: np.array, y_true) -> np.float64:
        """
        Returns the loss for predicted probabilities y_pred, with true class label y_true

        Args:
            y_pred (np.array): The predicted probabilities of each of the classes. 
                Array of shape (num_classes,) or (num_samples, num_classes).
            y_true: The true class label(s) of the data point(s). Could be a scalar or an array of shape (num_samples,).

        Returns:
            The (average) loss (across all samples), a np.float64 object.
        """
        return np.sum(np.zeros(1))

    def backward(self, y_pred: np.array, y_true) -> np.array:
        """
        Returns the derivative of the loss wrt the predicted probability of the true class y_true

        Args:
            y_pred (np.array): The predicted probabilities of each of the classes. 
                Array of shape (num_classes,) or (num_samples, num_classes).
            y_true: The true class label(s) of the data point(s). Could be a scalar or an array of shape (num_samples,).

        Returns:
            grad (np.array): The gradient of the loss wrt the inputs (probabilties) given to the loss function.
                Array of shape matching y_pred.
        """
        return y_pred

class CrossEntropyLoss(LossFunction):
    """
    Implements the cross entropy loss, L(y_hat, y) = -log(y_hat[y]) [where y is the true class label, from 0 to num_classes-1]
    """
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, y_pred: np.array, y_true) -> np.float64:
        """
        Returns the loss for predicted probabilities y_pred, with true class label y_true

        Args:
            y_pred (np.array): The predicted probabilities of each of the classes. 
                Array of shape (num_classes,) or (num_samples, num_classes).
            y_true: The true class label(s) of the data point(s). Could be a scalar or an array of shape (num_samples,).

        Returns:
            The (average) cross entropy loss (across all samples), a np.float64 object.
        """
        if len(y_pred.shape) == 1:
            return -np.log(y_pred[y_true])
        else:
            return np.mean(np.array([
                -np.log(y_pred[i][y]) for i,y in enumerate(y_true)
            ]))
    
    def backward(self, y_pred: np.array, y_true) -> np.array:
        """
        Returns the derivative of the loss wrt the predicted probability of the true class y_true

        Args:
            y_pred (np.array): The predicted probabilities of each of the classes. 
                Array of shape (num_classes,) or (num_samples, num_classes).
            y_true: The true class label(s) of the data point(s). Could be a scalar or an array of shape (num_samples,).

        Returns:
            grad (np.array): The gradient of the loss wrt the inputs (probabilties) given to the loss function.
                Array of shape matching y_pred.
        """
        grad = np.zeros_like(y_pred)
        if len(y_pred.shape) == 1:
            grad[y_true] = -1 / y_pred[y_true]
        else:
            for i,y in enumerate(y_true):
                grad[i][y] = -1 / y_pred[i][y]
        return grad

class MeanSquaredErrorLoss(LossFunction):
    """
    Implements the mean squared error loss, L(y_hat, y) = 0.5 * sum((y_hat - y)**2)
    """
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, y_pred: np.array, y_true) -> np.float64:
        """
        Returns the loss for predictions y_pred and true labels y_true.

        Args:
            y_pred (np.array): The predicted labels, array of shape (num_outputs,) or (num_samples, num_outputs).
            y_true (np.array or scalar): The true class labels, scalar or array of shape (num_samples,).

        Returns:
            The mean squared error across all samples, a np.float64 object.
        """
        y_true_proba = np.zeros_like(y_pred)
        if len(y_pred.shape) == 1:
            y_true_proba[y_true] == 1.
            count = 1
        else:
            for i,y in enumerate(y_true):
                y_true_proba[i][y] = 1.
            count = len(y_true)
        return 0.5 * np.sum((y_pred - y_true_proba)**2) / count

    def backward(self, y_pred: np.array, y_true) -> np.array:
        """
        Returns the derivative of the loss wrt the predicted probability of the true class y_true

        Args:
            y_pred (np.array): The predicted probabilities of each of the classes. 
                Array of shape (num_classes,) or (num_samples, num_classes).
            y_true: The true class label(s) of the data point(s). Could be a scalar or an array of shape (num_samples,).

        Returns:
            grad (np.array): The gradient of the loss wrt the inputs (probabilties) given to the loss function.
                Array of shape matching y_pred.
        """
        y_true_proba = np.zeros_like(y_pred)
        if len(y_pred.shape) == 1:
            y_true_proba[y_true] == 1.
        else:
            for i,y in enumerate(y_true):
                y_true_proba[i][y] = 1.
        return y_pred - y_true_proba
