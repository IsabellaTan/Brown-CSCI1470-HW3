from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
 
from typing import TypedDict, Dict, Union, Any, Optional, List, Iterable


class Tensor(np.ndarray):
    """
    Essentially, a NumPy Array with custom attriutes for gradient tracking
    Custom Tensor class that mimics torch.Tensor. Allows the ability for a numpy array to be marked as trainable.
    """

    requires_grad = True  ## Class variable; accessible by Tensor.requires_grad
    debug_print = False

    def __new__(cls, input_array, name: Optional[str] = None):
        if isinstance(input_array, Tensor):
            # If the input is already a Tensor, return it as is
            return input_array
        
        # Task 1: input the data to construct the object. 
        obj = np.asarray(a=???).view(type=cls)
        obj.backward = lambda x: None   ## Backward starts as None, gets assigned later
        obj.grad = None                 ## Gradient starts as None, gets computed later
        obj.requires_grad = True        ## By default, we'll want to compute gradient for new tensors
        obj.to = lambda x: obj          ## We don't handle special device support (i.e. cpu vs gpu/cuda)
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.backward       = getattr(obj, 'backward',      lambda x: None)
        self.to             = getattr(obj, 'to',            lambda x: obj)
        self.grad           = getattr(obj, 'grad',          None)
        self.requires_grad  = getattr(obj, 'requires_grad', None)
        self.name           = getattr(obj, 'name', None)

    def get_name(self) -> str:
        return self.name if self.name else self.__class__.__name__

    def __str__(self):
        if self.name is not None:
            return f"{super.__str__(self)[:-1]}, '{self.name}')"
        return super.__str__(self)

    def assign(self, value: Union[Tensor, np.ndarray]):
        self[:] = value

    def assign(self, value: Union[Tensor, np.ndarray]):
        """Assigns a new value to the tensor in-place"""
        self[:] = value

    class no_grad():

        '''
        Synergizes with Tensor: By entering the with scope of a no_grad object, 
        the Tensor.requires_grad singleton will swap to False. 
        '''

        def __enter__(self):
            # When tape scope is entered, let Diffable start recording to self.operation
            Tensor.requires_grad = False
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            # When tape scope is exited, stop asking Tensors to allow gradients
            Tensor.requires_grad = True


class Named:
    def __init__(self, name: Optional[str] = None):
        self.name = name

    def __str__(self):
        return self.name if self.name else self.__class__.__name__


class Callable(Named, ABC):
    """
    Modules that can be called like functions.
    """
    def __init__(self, *args, **kwargs):
        name = kwargs.get("name", None)
        super().__init__(name=name)

    def __call__(self, *args, **kwargs) -> Tensor:
        """
        Ensures `self()` and `self.forward()` be the same

        NOTE: This behavior can (and will) be overridden by Callable subclasses, 
                    in particular, the `Diffable` class
        """
        return Tensor(self.forward(*args, **kwargs))

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """
        Pass inputs through function.
        """
        pass


class Weighted(ABC):
    """
    Modules that have weights.
    """

    @property
    @abstractmethod
    def parameters(self) -> List[Tensor]:
        pass

    @property
    def trainable_variables(self) -> list[Tensor]:
        """Collects all trainable variables in the module"""
        return NotImplementedError

    @property
    def non_trainable_variables(self) -> list[Tensor]:
        """Collects all non-trainable variables in the module"""
        return NotImplementedError

    @property
    def trainable(self) -> bool:
        """Returns true if any of the weights are trainable"""
        return NotImplementedError

    @trainable.setter
    def trainable(self, trainable: bool):
        """Sets the trainable status of all weights to trainable"""
        pass 


class Diffable(Callable, Weighted, ABC):
    """
    Modules that keep track of gradients
    """

    def to(self, device):
        return self         # Just there to ignore device setting calls

    def __call__(self, *args, **kwargs) -> Tensor | list[Tensor]:
        """
        If there is a gradient tape scope in effect, perform AND RECORD the operation.
        Otherwise... just perform the operation and don't let the gradient tape know.
        """

        """ This uses some fancy python to grab all the information we need to perform the forward and
            backward passes. There are no TODOs in this class but understanding it will help with other
            parts of the assignment. It's a little confusing, so let's walk through it step by step.

            Abstractly, we need to do the following:
                1. Collect all the input values and their variable names
                2. Call the forward function with the input values
                3. If we should be recording gradients, 
                    Assign the output value's previous_layer field to be this diffable layer
                4. Return the output value(s)
        """

        """Start Task 1: Collect all the input values and their variable names"""

        # This line grabs the variable names that are passed to the call function, ingores self
        self.argnames = self.forward.__code__.co_varnames[1:]

        # Assigns the values passed in to the argnames we just grabbed
        ##  It's helpful to think of this line as a dictionary turning the unnamed "args" into named "kwargs"
        named_args = {self.argnames[i]: args[i] for i in range(len(args))}

        # Combines the unnamed args with the named args
        self.ins = {**named_args, **kwargs}

        # Grabs the input values of all passed args/kwargs from the constucted input dictionary
        self.inputs = list(self.ins.values())

        """End Task 1. and start Task 2: Call the forward function with the input values"""

        # Calls the forward function with the input values
        layer_output = self.forward(*args, **kwargs)
        if isinstance(layer_output, Tensor): layer_output.name = f"{self}:out"
        ## If there is only one output, make it a list so we can iterate over it
        self.outputs = [layer_output] if not isinstance(layer_output, list) else layer_output


        """End Task 2. and start Task 3: If we should be recording gradients,
                                        Assign the output value's previous_layer field to be this diffable layer"""

        ## It also makes sure that all of the outputs are differentiable, and 
        ## sets the backward method of these to point to the layer's backward
        out_list = [self.outputs] if hasattr(self.outputs, 'backward') else self.outputs
        for out in out_list:
            out.backward = self.backward


        """End Task 3. and start Task 4: Return the output value(s)"""
        return layer_output

    @property
    def parameters(self):
        """Returns a list of parameters"""
        return []

    @abstractmethod
    def forward(self, x):
        """Pass inputs through function. Can store inputs and outputs as instance variables"""
        pass

    @abstractmethod
    def get_input_gradients(self) -> list[Tensor]:
        """
        NOTE: required for all Diffable modules
        returns:
            list of gradients with respect to the inputs
        """
        return []

    @abstractmethod
    def get_weight_gradients(self) -> list[Tensor]:
        """
        NOTE: required for SOME Diffable modules
        returns:
            list of gradients with respect to the weights
        """
        return []

    def compose_input_gradients(self, J: Optional[List[Tensor]] = None) -> List[Tensor]:
        """
        Compose the inputted cumulative jacobian with the input jacobian for the layer.
        Implemented with batch-level vectorization.

        Requires `input_gradients` to provide either batched or overall jacobian.
        Assumes input/cumulative jacobians are matrix multiplied

        Note: This and compose_to_weight are generalized to handle a wide array of architectures
                so it handles a lot of edge cases that you may not need to worry about for this
                assignment. That being said, it's very close to how this really works in Tensorflow and
                it's helps A LOT to understand how this works so you can debug the gradient method.
        """
        # If J[0] is None, then we have no upstream gradients to compose with
        #  so we just return the input gradients
        if J is None or J[0] is None:
            return self.get_input_gradients()
        # J_out stores all input gradients to be tracked in backpropagation.
        J_out = []
        for upstream_jacobian in J:
            batch_size = upstream_jacobian.shape[0]
            for layer_input, inp_grad in zip(self.inputs, self.get_input_gradients()):
                if Tensor.debug_print: print(f'\n∂{self}/∂{layer_input.get_name()}: local = {inp_grad.shape}{f" and upstream = {upstream_jacobian.shape}"}')
                j_wrt_lay_inp = np.zeros(layer_input.shape, dtype=inp_grad.dtype)
                for sample in range(batch_size):
                    s_grad = inp_grad[sample] if len(inp_grad.shape) == 3 else inp_grad
                    try:
                        j_wrt_lay_inp[sample] = s_grad @ upstream_jacobian[sample]
                    except ValueError as e:
                        raise ValueError(
                            f"Error occured trying to perform `g_b @ j[b]` with {s_grad.shape} and {upstream_jacobian[sample].shape}:\n{e}"
                        )
                J_out += [j_wrt_lay_inp]
        # Returns cumulative jacobians w.r.t to all inputs.
        return J_out

    def compose_weight_gradients(self, J: Optional[List[Tensor]] = None) -> List[Tensor]:
        """
        Compose the inputted cumulative jacobian with the weight jacobian for the layer.
        Implemented with batch-level vectorization.

        Requires `weight_gradients` to provide either batched or overall jacobian.
        Assumes weight/cumulative jacobians are element-wise multiplied (w/ broadcasting)
        and the resulting per-batch statistics are averaged together for avg per-param gradient.
        """
        # Returns weights gradients if no apriori cumulative jacobians are provided.
        if J is None or J[0] is None:
            return self.get_weight_gradients()
        # J_out stores all weight gradients to be tracked in further backpropagation.
        J_out = []
        ## For every weight/weight-gradient pair...
        for upstream_jacobian in J:
            for layer_w, w_grad in zip(self.parameters, self.get_weight_gradients()):
                if Tensor.debug_print: print(f'\n∂{self}/∂{layer_w.get_name()}: local = {w_grad.shape}{f" and upstream = {upstream_jacobian.shape}"}')
                batch_size = upstream_jacobian.shape[0]
                ## Make a cumulative jacobian which will contribute to the final jacobian
                j_wrt_lay_w = np.zeros((batch_size, *layer_w.shape), dtype=w_grad.dtype)
                ## For every element in the batch (for a single batch-level gradient updates)
                for sample in range(batch_size):
                    ## If the weight gradient is a batch of transform matrices, get the right entry.
                    ## Allows gradient methods to give either batched or non-batched matrices
                    s_grad = w_grad[sample] if len(w_grad.shape) == 3 else w_grad
                    ## Update the batch's Jacobian update contribution
                    try:
                        ## this is a short cut that is equivalent to standard matmul composition
                        ## when using our shortcut linear layer gradient -> see note in layers for more details
                        j_wrt_lay_w[sample] = s_grad * upstream_jacobian[sample]
                    except ValueError as e:
                        raise ValueError(
                            f"Error occured trying to perform `g_b * j[b]` with {s_grad.shape} and {upstream_jacobian[sample].shape}:\n{e}"
                        )
                ## The final jacobian for this weight is the average gradient update for the batch
                J_out += [np.sum(j_wrt_lay_w, axis=0)]
            ## After new jacobian is computed for each weight set, return the list of gradient updatates
        return J_out

    def backward(self, grad: Optional[List[Tensor]] = None):
        """
        Propagate upstream gradient backwards by composing with local gradient
        """

        if Tensor.debug_print: print(f"Backwards on {self}")

        # TODO: Implement backward pass
        
        raise NotImplementedError