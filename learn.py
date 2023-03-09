from typing import Dict, List
from losses import *
from metrics import *
from nn import *
from optimizers import *

def fit(
    model: FeedForwardNeuralNetwork,
    X_train: np.array,
    Y_train,
    X_val: np.array,
    Y_val,
    optimizer: Optimizer,
    loss_fn: LossFunction = CrossEntropyLoss(),
    metric = categorical_accuracy,
    epochs: int = 10,
    batch_size: int = 64,
    verbose: int = 2
) -> Dict[str, List]:
    """
    Trains the model `model` on training data (X_train, Y_train) for `epochs` epochs using optimization algorithm `optimizer`,
    loss function `loss_fn`, and batch size `batch_size`, computing train and validation loss and score, using provided scoring
    metric `metric`.

    Args:
        model (FeedForwardNeuralNetwork): The model to train.
        X_train (np.array): Training data. Array of shape (num_samples_train, num_inputs) or (num_inputs,).
        Y_train (np.array or scalar): Training label(s). Array of shape (num_samples_train,) or int.
        X_val (np.array): Validation data. Array of shape (num_samples_val, num_inputs) or (num_inputs,).
        Y_val (np.array or scalar): Validation label(s). Array of shape (num_samples_val,) or int.
        optimizer (Optimizer): Optimization algorithm to use.
        loss_fn (LossFunction, optional): The loss function to use. Defaults to CrossEntropyLoss().
        metric (callable, optional): The scoring metric to use. Defaults to categorical_accuracy.
        epochs (int, optional): The number of epochs to train for. Defaults to 10.
        batch_size (int, optional): The batch size to use for training. Defaults to 64.
        verbose (int, optional): The verbosity of the information printed while training.
            verbose <= 0: Nothing is printed.
            0 < verbose <= 1: The final training and validation scores are printed.
            verbose > 1: Training and validation loss and score are printed for each epoch.
    
    Returns:
        history (dict[str, list]): A dictionary containing information on time/training evolution of the model.
            Valid keys: epoch, train_loss, train_score, val_loss, val_score.
            Each value is a list containing `epochs` entries, denoting the performance of the model during each
            epoch of training.
    """
    history = {
        'epoch': [],
        'train_loss': [],
        'train_score': [],
        'val_loss': [],
        'val_score': []
    }
    NUM_BATCHES = int(np.ceil(len(X_train)/batch_size))
    for epoch in range(epochs):
        train_loss = 0.0
        train_score = 0.0
        for batch in range(NUM_BATCHES):
            X_train_batch = X_train[batch*batch_size:np.minimum(batch_size*(batch+1), len(X_train))]
            Y_train_batch = Y_train[batch*batch_size:np.minimum(batch_size*(batch+1), len(X_train))]
            model.zero_grad()
            Y_pred_train = model.forward(X_train_batch)
            loss = loss_fn.forward(Y_pred_train, Y_train_batch)
            loss_grad = loss_fn.backward(Y_pred_train, Y_train_batch)
            model.backward(loss_grad, Y_train_batch)
            optimizer.step()
            train_loss += loss / NUM_BATCHES
            train_score += metric(Y_train_batch, Y_pred_train) / NUM_BATCHES
        Y_pred_val = model.forward(X_val, True)
        val_loss = loss_fn.forward(Y_pred_val, Y_val)
        val_score = metric(Y_val, Y_pred_val)
        history['epoch'].append(epoch+1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_score'].append(train_score)
        history['val_score'].append(val_score)
        if verbose > 1:
            print(f'[Epoch {epoch+1}/{epochs}] train loss: {train_loss:.6f}, val loss: {val_loss: .6f} || train score: {train_score:.6f}, val score: {val_score:.6f}')
    if verbose > 0:
        print(f'Scores at the end of training:\nTrain: {history["train_score"][-1]}\nValidation: {history["val_score"][-1]}')
    return history

def predict(
    model: FeedForwardNeuralNetwork,
    X: np.array
) -> np.array:
    """
    Predicts the outputs corresponding to provided data, as produced by the model.

    Args:
        model (FeedForwardNeuralNetwork): The model to predict with.
        X (np.array): The data to predict on. Array of shape (num_samples, num_inputs) or (num_inputs,).
    
    Returns:
        Y (np.array): The predicted probabilities of each class, corresponding to each data point.
            Array of shape (num_samples, num_outputs) or (num_outputs,).
    """
    return model.forward(X, True) 
    # eval_mode=True, so as to not store inputs, pre-activations and outputs for grad computation