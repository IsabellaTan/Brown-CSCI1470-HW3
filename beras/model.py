from abc import abstractmethod
from collections import defaultdict
from typing import Union, Dict, List, Tuple, Any, Optional

from beras.core import Diffable, Tensor, Callable
import numpy as np
import wandb

def print_stats(stat_dict:dict, batch_num=None, num_batches=None, epoch=None, avg=False):
    """
    Given a dictionary of names statistics and batch/epoch info,
    print them in an appealing manner. If avg, display stat averages.

    :param stat_dict: dictionary of metrics to display
    :param batch_num: current batch number
    :param num_batches: total number of batches
    :param epoch: current epoch number
    :param avg: whether to display averages
    """
    title_str = " - "
    if epoch is not None:
        title_str += f"Epoch {epoch+1:2}: "
    if batch_num is not None:
        title_str += f"Batch {batch_num+1:3}"
        if num_batches is not None:
            title_str += f"/{num_batches}"
    if avg:
        title_str += f"Average Stats"
    print(f"\r{title_str} : ", end="")
    op = np.mean if avg else lambda x: x
    print({k: np.round(op(v), 4) for k, v in stat_dict.items()}, end="")
    print("   ", end="" if not avg else "\n")


def update_metric_dict(super_dict: dict, sub_dict: dict):
    """
    Appends the average of the sub_dict metrics to the super_dict's metric list

    :param super_dict: dictionary of metrics to append to
    :param sub_dict: dictionary of metrics to average and append
    """
    for k, v in sub_dict.items():
        super_dict[k] += [np.mean(v)]


class Model(Diffable):

    def __init__(self, *args):
        """
        Initialize all trainable parameters and take layers as inputs
        """
        # Initialize all trainable parameters
        if len(args) == 1 and isinstance(args[0], list):
            self.layers = args[0]
        else:
            self.layers = list(args)

    def __call__(self, *args, **kwargs):
        """
        NOTE: DO NOT CHANGE
        Override Diffable call so that we don't actually have to
        implement backwards for the model
        And instead implicitly call backwards down the model stack
        """
        return self.forward(*args, **kwargs)

    @property
    def parameters(self) -> list[Tensor]:
        """
        Return the weights of the model by iterating through the layers
        """
        params = []
        for layer in self.layers:
            params.extend(layer.parameters)
        return params

    def compile(self, optimizer: Diffable, loss_fn: Diffable, acc_fn: Callable):
        """
        "Compile" the model by taking in the optimizers, loss, and accuracy functions.
        In more optimized DL implementations, this will have more involved processes
        that make the components extremely efficient but very inflexible.
        """
        self.optimizer      = optimizer
        self.compiled_loss  = loss_fn
        self.compiled_acc   = acc_fn

    def fit(self, 
            x: Tensor, y: Union[Tensor, np.ndarray], 
            epochs: int, batch_size: int, wandb_run: Optional[wandb.Run] = None) -> Dict[str, List[float]]:
        """
        Trains the model by iterating over the input dataset in batches and feeding input batches
        into the batch_step method with training. At the end, the metrics are returned.
        """
        history = defaultdict(list)
        num_samples = x.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))

        for epoch in range(epochs):

            batch_metrics = defaultdict(list)

            for batch in range(num_batches):

                start = batch * batch_size
                end = min(start + batch_size, num_samples)

                x_batch = x[start:end]
                y_batch = y[start:end]

                metrics = self.batch_step(x_batch, y_batch, training=True)

                for k, v in metrics.items():
                    batch_metrics[k].append(v)

                # print_stats(batch_metrics, batch, num_batches, epoch)

            update_metric_dict(history, batch_metrics)
            print_stats(history, epoch=epoch, avg=True)

        return history

    def evaluate(self, 
                 x: Tensor, y: Union[Tensor, np.ndarray], 
                 batch_size: int, wandb_run: Optional[wandb.Run] = None) -> Tuple[Dict[str, float], np.ndarray]:
        """
        X is the dataset inputs, Y is the dataset labels.
        Evaluates the model by iterating over the input dataset in batches and feeding input batches
        into the batch_step method. At the end, the metrics are returned. Should be called on
        the testing set to evaluate accuracy of the model using the metrics output from the fit method.

        NOTE: This method is almost identical to fit (think about how training and testing differ --
        the core logic should be the same)
        """
        metrics = defaultdict(list)
        predictions_list = []

        num_samples = x.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))

        for batch in range(num_batches):

            start = batch * batch_size
            end = min(start + batch_size, num_samples)

            x_batch = x[start:end]
            y_batch = y[start:end]

            batch_metrics = self.batch_step(x_batch, y_batch, training=False)

            preds = self.forward(x_batch)
            predictions_list.append(np.array(preds))

            for k, v in batch_metrics.items():
                metrics[k].append(v)

        avg_metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
        predictions = np.concatenate(predictions_list, axis=0)

        return avg_metrics, predictions

    def get_input_gradients(self) -> list[Tensor]:
        return super().get_input_gradients()

    def get_weight_gradients(self) -> list[Tensor]:
        return super().get_weight_gradients()
    
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass through the model - to be implemented by subclasses"""
        pass
    
    def backward(self, grad=np.array([[1]])):
        """Backward pass through the model - handled by layers"""
        pass
    
    @abstractmethod
    def batch_step(self, x: Tensor, y: Tensor, training:bool = True) -> dict[str, float]:
        """
        Computes loss and accuracy for a batch. This step consists of both a forward and backward pass.
        If training=false, don't apply gradients to update the model! 
        """
        raise NotImplementedError("batch_step method must be implemented in child class")

class Sequential(Model):
    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass in sequential model. 
        It's helpful to note that layers are initialized in beras.Model, and
        you can refer to them with self.layers. You can call a layer by doing var = layer(input).
        """
        if not isinstance(inputs, Tensor):
            inputs = Tensor(inputs)

        out = inputs

        for layer in self.layers:
            out = layer(out)
        return out

    def batch_step(self, x: Tensor, y: Tensor, training: bool = True) -> dict[str, float]:
        """
        Computes loss and accuracy for a batch. This step consists of a forward pass and (potentially)
        a backward pass and optimizer step. If training=false, don't apply gradients to update the model!
        """
        ## TODO: Compute loss and accuracy for a batch. Return as a dictionary
        ## If training, then also update the gradients according to the optimizer
    
        #if training:
        #    return {"loss": 0.0, "acc": 0.0}
        #else:
        #    return {"loss": 0.0, "acc": 0.0}, predictions
        if training:
            self.optimizer.zero_grad()
            
        predictions = self.forward(x)

        loss = self.compiled_loss(predictions, y)
        acc  = self.compiled_acc(predictions, y)

        if training:
            loss.backward()
            # self.optimizer.step(self.parameters)
            self.optimizer.step()

        return {"loss": float(loss), "acc": float(acc)}



