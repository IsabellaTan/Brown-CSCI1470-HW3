import numpy as np

from beras.core import Diffable, Tensor

class Activation(Diffable):
    @property
    def weights(self): return []

    def get_weight_gradients(self): return []


################################################################################
## Intermediate Activations To Put Between Layers

class LeakyReLU(Activation):

    ## TODO: Implement for default intermediate activation.

    def __init__(self, alpha: float = 0.3, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha

    def forward(self, x) -> Tensor:
        """Leaky ReLu forward propagation!"""
        out = np.where(x > 0, x, self.alpha * x)
        return Tensor(out)

    def get_input_gradients(self) -> list[Tensor]:
        """
        Leaky ReLu backpropagation!
        To see what methods/variables you have access to, refer to the cheat sheet.
        Hint: Make sure not to mutate any instance variables. Return a new list[tensor(s)]
        """
        x = self.inputs[0]
        grad = np.where(x > 0, 1, self.alpha)
        grad[x == 0] = 0
        return [grad]

    def compose_input_gradients(self, J):
        return self.get_input_gradients()[0] * J

class ReLU(LeakyReLU):
    ## GIVEN: Just shows that relu is a degenerate case of the LeakyReLU
    def __init__(self, **kwargs) -> None:
        super().__init__(alpha=0, **kwargs)


################################################################################
## Output Activations For Probability-Space Outputs

class Sigmoid(Activation):
    
    ## TODO: Implement for default output activation to bind output to 0-1
    
    def forward(self, x) -> Tensor:
        out = 1 / (1 + np.exp(-x))
        return Tensor(out)

    def get_input_gradients(self) -> list[Tensor]:
        """
        To see what methods/variables you have access to, refer to the cheat sheet.
        Hint: Make sure not to mutate any instance variables. Return a new list[tensor(s)]
        """
        y = self.outputs[0]
        grad = y * (1 - y)
        return [grad]

    def compose_input_gradients(self, J):
        return self.get_input_gradients()[0] * J


class Softmax(Activation):
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

    ## TODO [2470]: Implement for default output activation to bind output to 0-1

    def forward(self, x):
        """Softmax forward propagation!"""
        ## Not stable version
        ## exps = np.exp(inputs)
        ## outs = exps / np.sum(exps, axis=-1, keepdims=True)

        ## HINT: Use stable softmax, which subtracts maximum from
        ## all entries to prevent overflow/underflow issues
        shifted = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(shifted)
        out = exps / np.sum(exps, axis=1, keepdims=True)
        return Tensor(out)

    def get_input_gradients(self):
        """Softmax input gradients!"""
        x, y = self.inputs + self.outputs
        bn, n = x.shape
        grad = np.zeros(shape=(bn, n, n), dtype=x.dtype)
        
        # TODO: Implement softmax gradient
        for i in range(bn):
            yi = y[i]
            outer = np.outer(yi, yi)
            diag = np.diag(yi)
            grad[i] = diag - outer

        return [grad]



