import numpy as np
from beras.core import Callable, Tensor


class CategoricalAccuracy(Callable):
    def forward(self, probs: Tensor, labels: Tensor) -> float:
        ## TODO: Compute and return the categorical accuracy of your model 
        ## given the output probabilities and true labels. 
        ## HINT: Argmax + boolean mask via '=='
         
        # Get predicted class indices
        pred_classes = np.argmax(probs, axis=1)
        # Get true class indices (labels are one-hot)
        true_classes = np.argmax(labels, axis=1)
        # Compare predictions with ground truth
        correct = pred_classes == true_classes

        # Return mean accuracy
        return np.mean(correct)
    








