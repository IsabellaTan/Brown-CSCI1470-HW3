import numpy as np
from typing import List

from beras.core import Diffable, Tensor

class Loss(Diffable):
    @property
    def parameters(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return NotImplementedError

    def get_input_gradients(self) -> list[Tensor]:
        return NotImplementedError

class CategoricalCrossEntropy(Loss):
    def __init__(self, epsilon: float = 1e-12, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def forward(self, y_pred, y_true) -> Tensor:
        """
        Categorical cross entropy forward pass!
        Ensure we don't get log calculation errors by 
        clipping the values of y_pred to be between eps, 1-eps 
        (see np.clip)
        """
        return NotImplementedError

    def get_input_gradients(self) -> List[Tensor]:
        """Categorical cross entropy input gradient method!"""
        return NotImplementedError
