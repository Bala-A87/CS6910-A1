import numpy as np
from nn import FeedForwardNeuralNetwork
from typing import Tuple

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
        for layer in self.model.layers:
            layer.w -= self.lr * (layer.grad_w + self.weight_decay * layer.w)
            layer.b -= self.lr * (layer.grad_b + self.weight_decay * layer.b)

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
        self.u_ws = []
        self.u_bs = []
        for layer in model.layers:
            self.u_ws.append(np.zeros_like(layer.w))
            self.u_bs.append(np.zeros_like(layer.b))

    def step(self) -> None:
        """
        Performs one step/update to the parameters of the model, using grads computed by the model's backward function.
        """
        for layer, u_w, u_b in zip(self.model.layers, self.u_ws, self.u_bs):
            u_w = self.beta * u_w + self.lr * (layer.grad_w + self.weight_decay * layer.w)
            layer.w -= u_w
            u_b = self.beta * u_b + self.lr * (layer.grad_b + self.weight_decay * layer.b)
            layer.b -= u_b

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
        self.u_ws = []
        self.u_bs = []
        for layer in model.layers:
            self.u_ws.append(np.zeros_like(layer.w))
            self.u_bs.append(np.zeros_like(layer.b))
    
    def step(self) -> None:
        """
        Performs one step/update to the parameters of the model, using grads computed by the model's backward function.
        """
        for layer, u_w, u_b in zip(self.model.layers, self.u_ws, self.u_bs):
            u_w = self.beta * u_w + layer.grad_w
            layer.w -= self.lr * (self.beta * u_w + layer.grad_w + self.weight_decay * layer.w)
            u_b = self.beta * u_b + layer.grad_b
            layer.b -= self.lr * (self.beta * u_b + layer.grad_b + self.weight_decay * layer.b)

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
        self.v_ws = []
        self.v_bs = []
        for layer in model.layers:
            self.v_ws.append(np.zeros_like(layer.w))
            self.v_bs.append(np.zeros_like(layer.b))

    def step(self) -> None:
        """
        Performs one step/update to the parameters of the model, using grads computed by the model's backward function.
        """
        for layer, v_w, v_b in zip(self.model.layers, self.v_ws, self.v_bs):
            v_w = self.beta * v_w + (1 - self.beta) * layer.grad_w**2
            layer.w -= self.lr * (layer.grad_w + self.weight_decay * layer.w) / np.sqrt(v_w + self.eps)
            v_b = self.beta * v_b + (1 - self.beta) * layer.grad_b**2
            layer.b -= self.lr * (layer.grad_b + self.weight_decay * layer.b) / np.sqrt(v_b + self.eps)

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
        self.m_ws, self.v_ws = [], []
        self.m_bs, self.v_bs = [], []
        for layer in self.model.layers:
            self.m_ws.append(np.zeros_like(layer.w))
            self.v_ws.append(np.zeros_like(layer.w))
            self.m_bs.append(np.zeros_like(layer.b))
            self.v_bs.append(np.zeros_like(layer.b))

    def step(self) -> None:
        """
        Performs one step/update to the parameters of the model, using grads computed by the model's backward function.
        """
        for layer, m_w, v_w, m_b, v_b in zip(
            self.model.layers,
            self.m_ws,
            self.v_ws,
            self.m_bs,
            self.v_bs
        ):
            m_w = self.beta1 * m_w + (1 - self.beta1) * layer.grad_w
            v_w = self.beta2 * v_w + (1 - self.beta2) * layer.grad_w**2
            layer.w -= self.lr * ((m_w / (1 - self.pow_beta1)) / (self.eps + np.sqrt(v_w / (1 - self.pow_beta2))) + self.weight_decay * layer.w)
            m_b = self.beta1 * m_b + (1 - self.beta1) * layer.grad_b
            v_b = self.beta2 * v_b + (1 - self.beta2) * layer.grad_b**2
            layer.b -= self.lr * ((m_b / (1 - self.pow_beta1)) / (self.eps + np.sqrt(v_b / (1 - self.pow_beta2))) + self.weight_decay * layer.b)
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
        self.m_ws, self.v_ws = [], []
        self.m_bs, self.v_bs = [], []
        for layer in self.model.layers:
            self.m_ws.append(np.zeros_like(layer.w))
            self.v_ws.append(np.zeros_like(layer.w))
            self.m_bs.append(np.zeros_like(layer.b))
            self.v_bs.append(np.zeros_like(layer.b))

    def step(self) -> None:
        """
        Performs one step/update to the parameters of the model, using grads computed by the model's backward function.
        """
        self.pow_beta1 *= self.beta1
        self.pow_beta2 *= self.beta2
        for layer, m_w, v_w, m_b, v_b in zip(
            self.model.layers,
            self.m_ws,
            self.v_ws,
            self.m_bs,
            self.v_bs
        ):
            m_w = self.beta1 * m_w + (1 - self.beta1) * layer.grad_w
            v_w = self.beta2 * v_w + (1 - self.beta2) * layer.grad_w**2
            layer.w -= self.lr * ((self.beta1 * m_w + (1 - self.beta1) * layer.grad_w) /((1 - self.pow_beta1) * (self.eps + np.sqrt(v_w / (1 - self.pow_beta2)))) + self.weight_decay * layer.w)
            m_b = self.beta1 * m_b + (1 - self.beta1) * layer.grad_b
            v_b = self.beta2 * v_b + (1 - self.beta2) * layer.grad_b**2
            layer.b -= self.lr * ((self.beta1 * m_b + (1 - self.beta1) * layer.grad_b) /((1 - self.pow_beta1) * (self.eps + np.sqrt(v_b / (1 - self.pow_beta2)))) + self.weight_decay * layer.b)
