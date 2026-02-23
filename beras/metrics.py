import numpy as np

from beras.core import Callable, Tensor


class CategoricalAccuracy(Callable):
    def forward(self, probs: Tensor, labels: Tensor) -> float:
        ## TODO: Compute and return the categorical accuracy of your model 
        ## given the output probabilities and true labels. 
        ## HINT: Argmax + boolean mask via '=='
        return NotImplementedError
