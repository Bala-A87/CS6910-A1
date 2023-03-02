import numpy as np
from nn import FeedForwardNeuralNetwork

class Optimizer():
    """
    Template class to implement an optimization algorithm. Each child class must implement a `step` method, which uses the 
    computed gradients of the associated model to compute the updates to all the parameters of the network and perform an
    update step.

    Args:
        model (FeedForwardNeuralNetwork): The model whose parameters are to be learned.
        lr (float, optional): The learning rate for the updates. Defaults to 1e-3.
        weight_decay (float, optional): The L2-regularization to use for the parameters of the network.
            Defaults to 0.
    """
    def __init__(
            self,
            model: FeedForwardNeuralNetwork,
            lr: float = 1e-3,
            weight_decay: float = 0.
    ) -> None:
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
    
    def step(self) -> None:
        """
        Performs one step/update to the parameters of the model, using grads computed by the model's backward function.
        """
        pass

class StochasticGradientDescent(Optimizer):
    """
    Class to implement stochastic (mini-batch vanilla) gradient descent optimization algorithm on a model.

    Args:
        model (FeedForwardNeuralNetwork): The model whose parameters are to be learned.
        lr (float, optional): The learning rate for the updates. Defaults to 1e-3.
        weight_decay (float, optional): The L2-regularization to use for the parameters of the network.
            Defaults to 0.
    """
    def __init__(
        self,
        model: FeedForwardNeuralNetwork,
        lr: float = 1e-3,
        weight_decay: float = 0.
    ) -> None:
        super().__init__(model, lr, weight_decay)

    def step(self) -> None:
        """
        Performs one step/update to the parameters of the model, using grads computed by the model's backward function.
        """
        self.model.output_layer.w -= self.lr * (self.model.output_layer.grad_w + self.weight_decay * self.model.output_layer.w)
        self.model.output_layer.b -= self.lr * (self.model.output_layer.grad_b + self.weight_decay * self.model.output_layer.b)
        for hidden_layer in self.model.hidden_layers:
            hidden_layer.w -= self.lr * (hidden_layer.grad_w + self.weight_decay * hidden_layer.w)
            hidden_layer.b -= self.lr * (hidden_layer.grad_b + self.weight_decay * hidden_layer.b)
        self.model.input_layer.w -= self.lr * (self.model.input_layer.grad_w + self.weight_decay * self.model.input_layer.w)
        self.model.input_layer.b -= self.lr * (self.model.input_layer.grad_b + self.weight_decay * self.model.input_layer.b)

class MomentumGradientDescent(Optimizer):
    """
    Class to implement momentum-based gradient descent optimization algorithm on a model.

    Args:
        model (FeedForwardNeuralNetwork): The model whose parameters are to be learned.
        lr (float, optional): The learning rate for the updates. Defaults to 1e-3.
        beta (float, optional): The momentum to be used, quantifying the confidence in the history of updates.
            Should range from 0 to 1. Defaults to 0.9.
        weight_decay (float, optional): The L2-regularization to use for the parameters of the network.
            Defaults to 0.
    """
    def __init__(
        self,
        model: FeedForwardNeuralNetwork,
        lr: float = 1e-3,
        beta: float = 0.9,
        weight_decay: float = 0.
    ) -> None:
        super().__init__(model, lr, weight_decay)
        self.beta = beta
        self.input_u_w = np.zeros_like(model.input_layer.w)
        self.input_u_b = np.zeros_like(model.input_layer.b)
        self.hidden_u_ws = []
        self.hidden_u_bs = []
        for hidden_layer in model.hidden_layers:
            self.hidden_u_ws.append(np.zeros_like(hidden_layer.w))
            self.hidden_u_bs.append(np.zeros_like(hidden_layer.b))
        self.output_u_w = np.zeros_like(model.output_layer.w)
        self.output_u_b = np.zeros_like(model.output_layer.b)

    def step(self) -> None:
        """
        Performs one step/update to the parameters of the model, using grads computed by the model's backward function.
        """
        self.output_u_w = self.beta * self.output_u_w + self.lr * (self.model.output_layer.grad_w + self.weight_decay * self.model.output_layer.w)
        self.model.output_layer.w -= self.output_u_w
        self.output_u_b = self.beta * self.output_u_b + self.lr * (self.model.output_layer.grad_b + self.weight_decay * self.model.output_layer.b)
        self.model.output_layer.b -= self.output_u_b
        for hidden_layer, hidden_u_w, hidden_u_b in zip(self.model.hidden_layers, self.hidden_u_ws, self.hidden_u_bs):
            hidden_u_w = self.beta * hidden_u_w + self.lr * (hidden_layer.grad_w + self.weight_decay * hidden_layer.w)
            hidden_layer.w -= hidden_u_w
            hidden_u_b = self.beta * hidden_u_b + self.lr * (hidden_layer.grad_b + self.weight_decay * hidden_layer.b)
            hidden_layer.b -= hidden_u_b
        self.input_u_w = self.beta * self.input_u_w + self.lr * (self.model.input_layer.grad_w + self.weight_decay * self.model.input_layer.w)
        self.model.input_layer.w -= self.input_u_w
        self.input_u_b = self.beta * self.input_u_b + self.lr * (self.model.input_layer.grad_b + self.weight_decay * self.model.input_layer.b)
        self.model.input_layer.b -= self.input_u_b

class NesterovGradientDescent(Optimizer):
    """
    Class to implement Nesterov accelerated gradient descent optimization algorithm on a model.

    Args:
        model (FeedForwardNeuralNetwork): The model whose parameters are to be learned.
        lr (float, optional): The learning rate for the updates. Defaults to 1e-3.
        beta (float, optional): The momentum to be used, quantifying the confidence in the history of updates (and the lookahead).
            Should range from 0 to 1. Defaults to 0.9.
        weight_decay (float, optional): The L2-regularization to use for the parameters of the network.
            Defaults to 0.
    """
    def __init__(
        self,
        model: FeedForwardNeuralNetwork,
        lr: float = 1e-3,
        beta: float = 0.9,
        weight_decay : float = 0.
    ) -> None:
        super().__init__(model, lr, weight_decay)
        self.beta = beta
        self.input_u_w = np.zeros_like(model.input_layer.w)
        self.input_u_b = np.zeros_like(model.input_layer.b)
        self.hidden_u_ws = []
        self.hidden_u_bs = []
        for hidden_layer in model.hidden_layers:
            self.hidden_u_ws.append(np.zeros_like(hidden_layer.w))
            self.hidden_u_bs.append(np.zeros_like(hidden_layer.b))
        self.output_u_w = np.zeros_like(model.output_layer.w)
        self.output_u_b = np.zeros_like(model.output_layer.b)
    
    def step(self) -> None:
        """
        Performs one step/update to the parameters of the model, using grads computed by the model's backward function.
        """
        self.input_u_w = self.beta * self.input_u_w + self.model.input_layer.grad_w
        self.model.input_layer.w -= self.lr * (self.beta * self.input_u_w + self.model.input_layer.grad_w + self.weight_decay * self.model.input_layer.w)
        self.input_u_b = self.beta * self.input_u_b + self.model.input_layer.grad_b
        self.model.input_layer.b -= self.lr * (self.beta * self.input_u_b + self.model.input_layer.grad_b + self.weight_decay * self.model.input_layer.b)
        for hidden_layer, hidden_u_w, hidden_u_b in zip(self.model.hidden_layers, self.hidden_u_ws, self.hidden_u_bs):
            hidden_u_w = self.beta * hidden_u_w + hidden_layer.grad_w
            hidden_layer.w -= self.lr * (self.beta * hidden_u_w + hidden_layer.grad_w + self.weight_decay * hidden_layer.w)
            hidden_u_b = self.beta * hidden_u_b + hidden_layer.grad_b
            hidden_layer.b -= self.lr * (self.beta * hidden_u_b + hidden_layer.grad_b + self.weight_decay * hidden_layer.b)
        self.output_u_w = self.beta * self.output_u_w + self.model.output_layer.grad_w
        self.model.output_layer.w -= self.lr * (self.beta * self.output_u_w + self.model.output_layer.grad_w + self.weight_decay * self.model.output_layer.w)
        self.output_u_b = self.beta * self.output_u_b + self.model.output_layer.grad_b
        self.model.output_layer.b -= self.lr * (self.beta * self.output_u_b + self.model.output_layer.grad_b + self.weight_decay * self.model.output_layer.b)

class RMSProp(Optimizer):
    """
    Class to implement RMSProp optimization algorithm on a model.

    Args:
        model (FeedForwardNeuralNetwork): The model whose parameters are to be learned.
        lr (float, optional): The learning rate for the updates. Defaults to 1e-3.
        beta (float, optional): The momentum to be used, quantifying the confidence in the history of updates.
            Should range from 0 to 1. Defaults to 0.9.
        eps(float, optional): The amount to be added to the accumulated history, to add stability to computation.
            Defaults to 1e-8.
        weight_decay (float, optional): The L2-regularization to use for the parameters of the network.
            Defaults to 0.
    """
    def __init__(
            self,
            model: FeedForwardNeuralNetwork,
            lr: float = 1e-3,
            beta: float = 0.9,
            eps: float = 1e-8,
            weight_decay: float = 0.
    ) -> None:
        super().__init__(model, lr, weight_decay)
        self.beta = beta
        self.eps = eps
        self.input_v_w = np.zeros_like(model.input_layer.w)
        self.input_v_b = np.zeros_like(model.input_layer.b)
        self.hidden_v_ws = []
        self.hidden_v_bs = []
        for hidden_layer in model.hidden_layers:
            self.hidden_v_ws.append(np.zeros_like(hidden_layer.w))
            self.hidden_v_bs.append(np.zeros_like(hidden_layer.b))
        self.output_v_w = np.zeros_like(model.output_layer.w)
        self.output_v_b = np.zeros_like(model.output_layer.b)

    def step(self) -> None:
        """
        Performs one step/update to the parameters of the model, using grads computed by the model's backward function.
        """
        self.input_v_w = self.beta * self.input_v_w + (1 - self.beta) * self.model.input_layer.grad_w**2
        self.model.input_layer.w -= self.lr * (self.model.input_layer.grad_w + self.weight_decay * self.model.input_layer.w) / np.sqrt(self.input_v_w + self.eps)
        self.input_v_b = self.beta * self.input_v_b + (1 - self.beta) * self.model.input_layer.grad_b**2 
        self.model.input_layer.b -= self.lr * (self.model.input_layer.grad_b + self.weight_decay * self.model.input_layer.b) / np.sqrt(self.input_v_b + self.eps)
        for hidden_layer, hidden_v_w, hidden_v_b in zip(self.model.hidden_layers, self.hidden_v_ws, self.hidden_v_bs):
            hidden_v_w = self.beta * hidden_v_w + (1 - self.beta) * hidden_layer.grad_w**2
            hidden_layer.w -= self.lr * (hidden_layer.grad_w + self.weight_decay * hidden_layer.w) / np.sqrt(hidden_v_w + self.eps)
            hidden_v_b = self.beta * hidden_v_b + (1 - self.beta) * hidden_layer.grad_b**2
            hidden_layer.b -= self.lr * (hidden_layer.grad_b + self.weight_decay * hidden_layer.b) / np.sqrt(hidden_v_b + self.eps)
        self.output_v_w = self.beta * self.output_v_w + (1 - self.beta) * self.model.output_layer.grad_w**2
        self.model.output_layer.w -= self.lr * (self.model.output_layer.grad_w + self.weight_decay * self.model.output_layer.w) / np.sqrt(self.output_v_w + self.eps)
        self.output_v_b = self.beta * self.output_v_b + (1 - self.beta) * self.model.output_layer.grad_b**2 
        self.model.output_layer.b -= self.lr * (self.model.output_layer.grad_b + self.weight_decay * self.model.output_layer.b) / np.sqrt(self.output_v_b + self.eps)

from typing import Tuple

class Adam(Optimizer):
    """
    Class to implement Adam optimization algorithm on a model.

    Args:
        model (FeedForwardNeuralNetwork): The model whose parameters are to be learned.
        lr (float, optional): The learning rate for the updates. Defaults to 1e-3.
        betas (tuple[float, float], optional): The betas to be used to store the history, as in momentum and in 
            RMSProp. betas[0] corresponds to momentum, betas[1] corresponds to beta of RMSProp.
            Both values to range from 0 to 1. Defaults to (0.9, 0.999).
        eps(float, optional): The amount to be added to the accumulated history, to add stability to computation.
            Defaults to 1e-8.
        weight_decay (float, optional): The L2-regularization to use for the parameters of the network.
            Defaults to 0.
    """
    def __init__(
            self,
            model: FeedForwardNeuralNetwork,
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.
    ) -> None:
        super().__init__(model, lr, weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.pow_beta1, self.pow_beta2 = betas
        self.input_m_w, self.input_v_w = np.zeros_like(self.model.input_layer.w), np.zeros_like(self.model.input_layer.w)
        self.input_m_b, self.input_v_b = np.zeros_like(self.model.input_layer.b), np.zeros_like(self.model.input_layer.b)
        self.hidden_m_ws, self.hidden_v_ws = [], []
        self.hidden_m_bs, self.hidden_v_bs = [], []
        for hidden_layer in self.model.hidden_layers:
            self.hidden_m_ws.append(np.zeros_like(hidden_layer.w))
            self.hidden_v_ws.append(np.zeros_like(hidden_layer.w))
            self.hidden_m_bs.append(np.zeros_like(hidden_layer.b))
            self.hidden_v_bs.append(np.zeros_like(hidden_layer.b))
        self.output_m_w, self.output_v_w = np.zeros_like(self.model.output_layer.w), np.zeros_like(self.model.output_layer.w)
        self.output_m_b, self.output_v_b = np.zeros_like(self.model.output_layer.b), np.zeros_like(self.model.output_layer.b)

    def step(self) -> None:
        """
        Performs one step/update to the parameters of the model, using grads computed by the model's backward function.
        """
        self.input_m_w = self.beta1 * self.input_m_w + (1 - self.beta1) * self.model.input_layer.grad_w
        self.input_v_w = self.beta2 * self.input_v_w + (1 - self.beta2) * self.model.input_layer.grad_w**2
        self.model.input_layer.w -= self.lr * ((self.input_m_w / (1 - self.pow_beta1)) / (self.eps + np.sqrt(self.input_v_w / (1 - self.pow_beta2))) + self.weight_decay * self.model.input_layer.w)
        self.input_m_b = self.beta1 * self.input_m_b + (1 - self.beta1) * self.model.input_layer.grad_b
        self.input_v_b = self.beta2 * self.input_v_b + (1 - self.beta2) * self.model.input_layer.grad_b**2
        self.model.input_layer.b -= self.lr * ((self.input_m_b / (1 - self.pow_beta1)) / (self.eps + np.sqrt(self.input_v_b / (1 - self.pow_beta2))) + self.weight_decay * self.model.input_layer.b)
        for hidden_layer, hidden_m_w, hidden_v_w, hidden_m_b, hidden_v_b in zip(
            self.model.hidden_layers,
            self.hidden_m_ws,
            self.hidden_v_ws,
            self.hidden_m_bs,
            self.hidden_v_bs
        ):
            hidden_m_w = self.beta1 * hidden_m_w + (1 - self.beta1) * hidden_layer.grad_w
            hidden_v_w = self.beta2 * hidden_v_w + (1 - self.beta2) * hidden_layer.grad_w**2
            hidden_layer.w -= self.lr * ((hidden_m_w / (1 - self.pow_beta1)) / (self.eps + np.sqrt(hidden_v_w / (1 - self.pow_beta2))) + self.weight_decay * hidden_layer.w)
            hidden_m_b = self.beta1 * hidden_m_b + (1 - self.beta1) * hidden_layer.grad_b
            hidden_v_b = self.beta2 * hidden_v_b + (1 - self.beta2) * hidden_layer.grad_b**2
            hidden_layer.b -= self.lr * ((hidden_m_b / (1 - self.pow_beta1)) / (self.eps + np.sqrt(hidden_v_b / (1 - self.pow_beta2))) + self.weight_decay * hidden_layer.b)
        self.output_m_w = self.beta1 * self.output_m_w + (1 - self.beta1) * self.model.output_layer.grad_w
        self.output_v_w = self.beta2 * self.output_v_w + (1 - self.beta2) * self.model.output_layer.grad_w**2
        self.model.output_layer.w -= self.lr * ((self.output_m_w / (1 - self.pow_beta1)) / (self.eps + np.sqrt(self.output_v_w / (1 - self.pow_beta2))) + self.weight_decay * self.model.output_layer.w)
        self.output_m_b = self.beta1 * self.output_m_b + (1 - self.beta1) * self.model.output_layer.grad_b
        self.output_v_b = self.beta2 * self.output_v_b + (1 - self.beta2) * self.model.output_layer.grad_b**2
        self.model.output_layer.b -= self.lr * ((self.output_m_b / (1 - self.pow_beta1)) / (self.eps + np.sqrt(self.output_v_b / (1 - self.pow_beta2))) + self.weight_decay * self.model.output_layer.b)
        self.pow_beta1 *= self.beta1
        self.pow_beta2 *= self.beta2

class NAdam(Optimizer):
    """
    Class to implement NAdam optimization algorithm on a model.

    Args:
        model (FeedForwardNeuralNetwork): The model whose parameters are to be learned.
        lr (float, optional): The learning rate for the updates. Defaults to 1e-3.
        betas (tuple[float, float], optional): The betas to be used to store the history, as in momentum and in 
            RMSProp. betas[0] corresponds to momentum (and lookahead), betas[1] corresponds to beta of RMSProp.
            Both values to range from 0 to 1. Defaults to (0.9, 0.999).
        eps(float, optional): The amount to be added to the accumulated history, to add stability to computation.
            Defaults to 1e-8.
        weight_decay (float, optional): The L2-regularization to use for the parameters of the network.
            Defaults to 0.
    """
    def __init__(
            self,
            model: FeedForwardNeuralNetwork,
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.
    ) -> None:
        super().__init__(model, lr, weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.pow_beta1, self.pow_beta2 = betas
        self.input_m_w, self.input_v_w = np.zeros_like(self.model.input_layer.w), np.zeros_like(self.model.input_layer.w)
        self.input_m_b, self.input_v_b = np.zeros_like(self.model.input_layer.b), np.zeros_like(self.model.input_layer.b)
        self.hidden_m_ws, self.hidden_v_ws = [], []
        self.hidden_m_bs, self.hidden_v_bs = [], []
        for hidden_layer in self.model.hidden_layers:
            self.hidden_m_ws.append(np.zeros_like(hidden_layer.w))
            self.hidden_v_ws.append(np.zeros_like(hidden_layer.w))
            self.hidden_m_bs.append(np.zeros_like(hidden_layer.b))
            self.hidden_v_bs.append(np.zeros_like(hidden_layer.b))
        self.output_m_w, self.output_v_w = np.zeros_like(self.model.output_layer.w), np.zeros_like(self.model.output_layer.w)
        self.output_m_b, self.output_v_b = np.zeros_like(self.model.output_layer.b), np.zeros_like(self.model.output_layer.b)

    def step(self) -> None:
        """
        Performs one step/update to the parameters of the model, using grads computed by the model's backward function.
        """
        self.pow_beta1 *= self.beta1
        self.pow_beta2 *= self.beta2
        self.input_m_w = self.beta1 * self.input_m_w + (1 - self.beta1) * self.model.input_layer.grad_w
        self.input_v_w = self.beta2 * self.input_v_w + (1 - self.beta2) * self.model.input_layer.grad_w**2
        self.model.input_layer.w -= self.lr * ((self.beta1 * self.input_m_w + (1 - self.beta1) * self.model.input_layer.grad_w) /((1 - self.pow_beta1) * (self.eps + np.sqrt(self.input_v_w / (1 - self.pow_beta2)))) + self.weight_decay * self.model.input_layer.w)
        self.input_m_b = self.beta1 * self.input_m_b + (1 - self.beta1) * self.model.input_layer.grad_b
        self.input_v_b = self.beta2 * self.input_v_b + (1 - self.beta2) * self.model.input_layer.grad_b**2
        self.model.input_layer.b -= self.lr * ((self.beta1 * self.input_m_b + (1 - self.beta1) * self.model.input_layer.grad_b) /((1 - self.pow_beta1) * (self.eps + np.sqrt(self.input_v_b / (1 - self.pow_beta2)))) + self.weight_decay * self.model.input_layer.b)
        for hidden_layer, hidden_m_w, hidden_v_w, hidden_m_b, hidden_v_b in zip(
            self.model.hidden_layers,
            self.hidden_m_ws,
            self.hidden_v_ws,
            self.hidden_m_bs,
            self.hidden_v_bs
        ):
            hidden_m_w = self.beta1 * hidden_m_w + (1 - self.beta1) * hidden_layer.grad_w
            hidden_v_w = self.beta2 * hidden_v_w + (1 - self.beta2) * hidden_layer.grad_w**2
            hidden_layer.w -= self.lr * ((self.beta1 * hidden_m_w + (1 - self.beta1) * hidden_layer.grad_w) /((1 - self.pow_beta1) * (self.eps + np.sqrt(hidden_v_w / (1 - self.pow_beta2)))) + self.weight_decay * hidden_layer.w)
            hidden_m_b = self.beta1 * hidden_m_b + (1 - self.beta1) * hidden_layer.grad_b
            hidden_v_b = self.beta2 * hidden_v_b + (1 - self.beta2) * hidden_layer.grad_b**2
            hidden_layer.b -= self.lr * ((self.beta1 * hidden_m_b + (1 - self.beta1) * hidden_layer.grad_b) /((1 - self.pow_beta1) * (self.eps + np.sqrt(hidden_v_b / (1 - self.pow_beta2)))) + self.weight_decay * hidden_layer.b)
        self.output_m_w = self.beta1 * self.output_m_w + (1 - self.beta1) * self.model.output_layer.grad_w
        self.output_v_w = self.beta2 * self.output_v_w + (1 - self.beta2) * self.model.output_layer.grad_w**2
        self.model.output_layer.w -= self.lr * ((self.beta1 * self.output_m_w + (1 - self.beta1) * self.model.output_layer.grad_w) /((1 - self.pow_beta1) * (self.eps + np.sqrt(self.output_v_w / (1 - self.pow_beta2)))) + self.weight_decay * self.model.output_layer.w)
        self.output_m_b = self.beta1 * self.output_m_b + (1 - self.beta1) * self.model.output_layer.grad_b
        self.output_v_b = self.beta2 * self.output_v_b + (1 - self.beta2) * self.model.output_layer.grad_b**2
        self.model.output_layer.b -= self.lr * ((self.beta1 * self.output_m_b + (1 - self.beta1) * self.model.output_layer.grad_b) /((1 - self.pow_beta1) * (self.eps + np.sqrt(self.output_v_b / (1 - self.pow_beta2)))) + self.weight_decay * self.model.output_layer.b)
