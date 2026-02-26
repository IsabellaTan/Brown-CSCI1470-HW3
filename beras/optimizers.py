import numpy as np
from typing import List
from abc import ABC, abstractmethod

from beras.core import Tensor

class Optimizer(ABC):
    def __init__(self, parameters: List[Tensor], learning_rate: float):
        self.learning_rate = learning_rate
        self.parameters = parameters
    
    def zero_grad(self):
        for param in self.parameters:
            param.grad = None

    @abstractmethod
    def step(self):
        raise NotImplementedError("Optimizer step method must be implemented in child class")


class BasicOptimizer(Optimizer):
    
    def __init__(self, parameters: List[Tensor], learning_rate: float):
        super().__init__(parameters, learning_rate)

    def step(self):
        for param in self.parameters:
            if param.requires_grad and param.grad is not None:
                param -= self.learning_rate * param.grad


class RMSProp(Optimizer):

    def __init__(self, parameters: List[Tensor], learning_rate: float, beta: float = 0.9, epsilon: float = 1e-6):
        super().__init__(parameters, learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.v = [0 for _ in range(len(parameters))]

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.requires_grad and param.grad is not None:
                # Update moving average of squared gradients
                self.v[i] = self.beta * self.v[i] + (1 - self.beta) * (param.grad ** 2)
                # Update parameter
                param -= self.learning_rate * param.grad / (np.sqrt(self.v[i]) + self.epsilon)


class Adam(Optimizer):

    def __init__(
        self, 
        parameters: List[Tensor], 
        learning_rate: float, 
        beta_1: float = 0.9, 
        beta_2: float = 0.999, 
        epsilon: float = 1e-7
    ):
        super().__init__(parameters, learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        
        self.m = [0 for _ in range(len(parameters))] # First moment zero vector
        self.v = [0 for _ in range(len(parameters))] # Second moment zero vector.
        self.t = 0                                   # Time counter

    def step(self):
        # increment time step
        self.t += 1

        for i, param in enumerate(self.parameters):
            if param.requires_grad and param.grad is not None:

                g = param.grad

                # Update first moment
                self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * g

                # Update second moment
                self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * (g ** 2)

                # Bias correction
                m_hat = self.m[i] / (1 - self.beta_1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta_2 ** self.t)

                # Update parameter
                param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
