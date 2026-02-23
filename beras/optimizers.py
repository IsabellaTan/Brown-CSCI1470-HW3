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
        return NotImplementedError


class RMSProp(Optimizer):

    def __init__(self, parameters: List[Tensor], learning_rate: float, beta: float = 0.9, epsilon: float = 1e-6):
        super().__init__(parameters, learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.v = [0 for _ in range(len(parameters))]

    def step(self):
        return NotImplementedError


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
        return NotImplementedError
