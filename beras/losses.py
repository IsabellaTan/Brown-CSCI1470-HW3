import numpy as np
# due to Windows system
np.random.randint = lambda low, high=None, size=None: \
    np.random.mtrand._rand.randint(low, high, size).astype(np.int64)
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
        loss = np.mean(np.mean((y_true - y_pred) ** 2, axis=1))
        return Tensor(loss)

    def get_input_gradients(self) -> list[Tensor]:
        y_pred, y_true = self.inputs
        bn, n = y_pred.shape
        grad_pred = (2 / (bn * n)) * (y_pred - y_true)
        grad_true = np.zeros_like(y_true)

        return [grad_pred, grad_true]

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
        # clip predictions for numerical stability
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        # compute per-sample loss
        per_sample_loss = -np.sum(y_true * np.log(y_pred), axis=1)
        # average over batch
        loss = np.mean(per_sample_loss)

        return Tensor(loss)

    def get_input_gradients(self) -> List[Tensor]:
        """Categorical cross entropy input gradient method!"""
        y_pred, y_true = self.inputs
        batch_size = y_pred.shape[0]
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        # gradient w.r.t y_pred
        grad_pred = -(1 / batch_size) * (y_true / y_pred)

        # y_true treated as constant
        grad_true = np.zeros_like(y_true)

        return [grad_pred, grad_true]




