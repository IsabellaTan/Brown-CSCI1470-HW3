import numpy as np

from typing import Literal
from beras.core import Diffable, Tensor

LINEAR_INITIALIZERS = Literal["zero", "normal", "xavier", "kaiming", "xavier uniform", "kaiming uniform"]

class Linear(Diffable):

    def __init__(self, input_size: int, output_size: int, initializer: LINEAR_INITIALIZERS = "normal", **kwargs) -> None:
        super().__init__(**kwargs)
        self.w, self.b = self._initialize_weight(initializer, input_size, output_size)
        self.w.name, self.b.name = f"{str(self)}:W", f"{str(self)}:b"

    @property
    def parameters(self) -> list[Tensor]:
        return self.w, self.b

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for a dense layer! Refer to lecture slides for how this is computed.
        """
        return Tensor(x @ self.w + self.b)

    def get_input_gradients(self) -> list[Tensor]:
        #batch_size = self.inputs[0].shape[0]
        #input_size, output_size = self.w.shape
        #grad = np.zeros((batch_size, input_size, output_size))
        #for i in range(batch_size):
        #    grad[i] = self.w
        #return [grad]
        
        #x = self.inputs[0]
        #batch_size = x.shape[0]
        #input_size, output_size = self.w.shape
        #grad = np.zeros((batch_size, input_size, output_size))
        #for i in range(batch_size):
        #    grad[i] = self.w
        #return [grad]
    
        return [self.w]

    def get_weight_gradients(self) -> list[Tensor]:
        x = self.inputs[0]
        batch_size = x.shape[0]
        input_size, output_size = self.w.shape

        dW = np.zeros((batch_size, input_size, output_size))
        for i in range(batch_size):
            dW[i] = np.outer(x[i], np.ones(output_size))

        db = np.ones(output_size)

        return [dW, db]

    @staticmethod
    def _initialize_weight(initializer, input_size, output_size) -> tuple[Tensor, Tensor]:
        """
        Initializes the values of the weights and biases. The bias weights should always start at zero.
        However, the weights should follow the given distribution defined by the initializer parameter
        (zero, normal, xavier, or kaiming). You can do this with an if statement
        cycling through each option!

        Details on each weight initialization option:
            - Zero: Weights and biases contain only 0's. Generally a bad idea since the gradient update
            will be the same for each weight so all weights will have the same values.
            - Normal: Weights are initialized according to a normal distribution.
            - Xavier: Goal is to initialize the weights so that the variance of the activations are the
            same across every layer. This helps to prevent exploding or vanishing gradients. Typically
            works better for layers with tanh or sigmoid activation.
            - Kaiming: Similar purpose as Xavier initialization. Typically works better for layers
            with ReLU activation.
        """

        initializer = initializer.lower()
        assert initializer in (
            "zero",
            "normal",
            "xavier",
            "kaiming",
        ), f"Unknown dense weight initialization strategy '{initializer}' requested"

        if initializer == "zero":
            w = np.zeros((input_size, output_size))

        elif initializer == "normal":
            w = np.random.randn(input_size, output_size)

        elif initializer == "xavier":
            std = np.sqrt(2.0 / (input_size + output_size))
            w = np.random.randn(input_size, output_size) * std

        elif initializer == "kaiming":
            std = np.sqrt(2.0 / input_size)
            w = np.random.randn(input_size, output_size) * std

        else:
            raise ValueError(f"Unknown initializer: {initializer}")

        b = np.zeros(output_size)

        return Tensor(w), Tensor(b)
