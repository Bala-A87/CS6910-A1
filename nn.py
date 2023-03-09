import numpy as np
from activations import *

class Layer():
    """
    Models a single layer of a feedforward neural network. 

    Args:
        input_size (int): number of inputs to the layer
        output_size (int): number of outputs produced by the layer
        activation (str, optional): activation function used by the layer. 
            Allowed activations: 'identity', 'sigmoid', 'tanh', 'relu', 'softmax'
            Defaults to 'sigmoid'.
        weight_init (str, optional): type of initialization to be performed for the weights
            Allowed initializations: 'random', 'xavier'
            Defaults to 'random'.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation = 'sigmoid',
        weight_init: str = 'random',
        clip_norm: float = 1.
    ) -> None:
        if weight_init == 'xavier':
            self.w = np.random.normal(scale=np.sqrt(2/(input_size + output_size)), size=(output_size, input_size))
        else:
            self.w = np.random.normal(size=(output_size, input_size))   # shape: (output_size, input_size)
        self.b = np.zeros((output_size,))
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)
        if activation == 'identity' or activation.__class__ == Identity:
            self.activation = Identity()
        elif activation == 'sigmoid' or activation.__class__ == Sigmoid:
            self.activation = Sigmoid()
        elif activation == 'tanh' or activation.__class__ == Tanh:
            self.activation = Tanh()
        elif activation == 'relu' or activation.__class__ == ReLU:
            self.activation = ReLU()
        elif activation == 'softmax' or activation.__class__ == Softmax:
            self.activation = Softmax()
        self.clip_norm = clip_norm
    
    def forward(self, x: np.array, eval_mode: bool = False) -> np.array:
        """
        Computes the output of the layer for given input.

        Args: 
            x (np.array): the input vector for the layer, of size (input_size,) or (num_samples, input_size).
            eval_mode (bool, optional): whether to compute in eval mode, without storing input, pre-activation and output.
                Defaults to False.
        Returns: h (np.array): the output produced by the layer, of size (output_size,) or (num_samples, output_size).
        """
        if not eval_mode:
            self.input = x
            self.pre_activation = np.matmul(x, self.w.T) + self.b
            self.output = self.activation.forward(self.pre_activation)
            return self.output
        else:
            return self.activation.forward(np.matmul(x, self.w.T) + self.b)
    
    def backward(self, accumulated_grads: np.array, w_next: np.array = None) -> np.array:
        """
        Computes the relevant necessary gradients of the layer, given the gradient accumulated until the succeeding layer.

        Args:
            accumulated_grads (np.array): gradient accumulated from the loss until the following layer, i.e. grad(L) wrt a_(i+1)
                Array of shape (num_features,) or (num_samples, num_features).
            w_next (np.array, optional): weight associated with the succeeding layer. Not applicable for the output layer.
                Defaults to None.
        
        Returns: grad_to_return (np.array), gradient accumulated until the current layer, i.e., grad(L) wrt a_i
        """
        # accumulated_grads ~ grad(L) wrt a_i+1
        activation_grad = self.activation.backward(self.output)
        if w_next is not None:
            grad_to_return = np.matmul(accumulated_grads, w_next) * activation_grad
        else:
            if len(self.output.shape) == 1:
                grad_to_return = np.matmul(accumulated_grads.reshape(1, -1), activation_grad)
            else:
                grad_to_return = np.array([np.matmul(accumulated_grads[i].reshape(1, -1), activation_grad[i]).reshape(-1) for i in range(len(accumulated_grads))])
        # grad_to_return ~ grad(L) wrt a_i
        self.grad_w += np.matmul(grad_to_return.T.reshape(self.w.shape[0], -1), self.input.reshape(-1, self.w.shape[1])) / self.input.reshape(-1, self.w.shape[1]).shape[0]
        if len(self.input.shape) == 1:
            self.grad_b += grad_to_return
        else:
            self.grad_b += np.mean(grad_to_return, axis=0)
        if self.clip_norm is not None:
            norm_grad_w = np.sqrt(np.sum(self.grad_w**2)) / (self.w.shape[0] * self.w.shape[1])
            norm_grad_b = np.sqrt(np.sum(self.grad_b**2)) / self.b.shape[0]
            if norm_grad_w > self.clip_norm:
                self.grad_w /= norm_grad_w
            if norm_grad_b > self.clip_norm:
                self.grad_b /= norm_grad_b
        return grad_to_return

class FeedForwardNeuralNetwork():
    """
    Models a vanilla feedforward neural network, with softmax activation for the outputs.

    Args:
        num_layers (int): number of hidden layers in the network
        hidden_size (int): number of neurons per hidden layer
        input_size (int): size/dimension of inputs given to the network. 
            Defaults to 784, flattened size of mnist/fashion-mnist data.
        output_size (int): number of outputs produced by the network.
            Defaults to 10, number of classes of mnist/fashion-mnist data.
        activation (str, optional): activation function for the hidden layers.
            Allowed activations: 'identity', 'sigmoid', 'tanh', 'relu'
            Defaults to 'sigmoid'.
        weight_init (str, optional): type of initialization to be performed for the weights
            Allowed initializations: 'random', 'xavier'
            Defaults to 'random'.
    """
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        input_size: int = 784,
        output_size: int = 10,
        activation = 'sigmoid',
        weight_init: str = 'random',
        clip_norm: float = 1.
    ) -> None:
        self.input_layer = Layer(input_size, hidden_size, activation, weight_init, clip_norm)
        self.hidden_layers = []
        for i in range(num_layers-1):
            self.hidden_layers.append(Layer(hidden_size, hidden_size, activation, weight_init, clip_norm))
        self.output_layer = Layer(hidden_size, output_size, 'softmax', weight_init, clip_norm)
    
    def forward(self, x: np.array, eval_mode: bool = False) -> np.array:
        """
        Computes the output of the network for given input.

        Args: 
            x (numpy.array): the input to the network, of size (input_size,) or (num_samples, input_size)
            eval_mode (bool, optional): whether to compute in eval mode, without storing input, pre-activation and output
            in each layer. Defaults to False.
        Returns: y_hat (numpy.array): the output produced by the network, of size (output_size,) or (num_samples, output_size)
        """
        result = self.input_layer.forward(x, eval_mode)
        for hidden_layer in self.hidden_layers:
            result = hidden_layer.forward(result, eval_mode)
        return self.output_layer.forward(result, eval_mode)
    
    def backward(self, accumulated_grads: np.array, y_true) -> None:
        """
        Computes the gradient of the loss wrt all the parameters (weights, biases) of the network, given the gradient of the loss
        wrt the output of the network.

        Args:
            accumulated_grads (np.array): gradient of the loss wrt the outputs of the network
                Array of size (num_classes,) or (num_samples, num_classes).
            y_true: the true class label(s) for the data input(s) to the network
        """
        self.y_true = y_true
        accumulated_grads = self.output_layer.backward(accumulated_grads)
        w_next = self.output_layer.w
        for hidden_layer in reversed(self.hidden_layers):
            accumulated_grads = hidden_layer.backward(accumulated_grads, w_next=w_next)
            w_next = hidden_layer.w
        self.input_layer.backward(accumulated_grads, w_next=w_next)
    
    def zero_grad(self) -> None:
        """
        Sets the gradients of all the parameters of the network to zero.
        """
        self.output_layer.grad_w, self.output_layer.grad_b = 0., 0.
        for hidden_layer in self.hidden_layers:
            hidden_layer.grad_w, hidden_layer.grad_b = 0., 0.
        self.input_layer.grad_w, self.input_layer.grad_b = 0., 0.
