from types import SimpleNamespace
import numpy as np
import wandb

from preprocess import load_and_preprocess_data
from visualize import visualize_predictions

from beras.core import Tensor
from beras.layers import Linear
from beras.activations import ReLU, LeakyReLU, Softmax
from beras.losses import CategoricalCrossEntropy, MeanSquaredError
from beras.metrics import CategoricalAccuracy
from beras.onehot import OneHotEncoder
from beras.optimizers import BasicOptimizer, RMSProp, Adam
from beras.model import Model, Sequential

def get_model():
    model = Sequential(
        # Add in your layers here as elements of the list!
        # e.g. Dense(10, 10),
    )
    return model

def get_optimizer(model: Model):
    # choose an optimizer, initialize it and return it!
    return ...

def get_loss_fn():
    # choose a loss function, initialize it and return it!
    return ...

def get_acc_fn():
    # choose an accuracy metric, initialize it and return it!
    return ...

if __name__ == '__main__':

    ### Use this area to test your implementation!
    Tensor.debug_print = False #turn to True for extra print statements during compose!

    # 0. Optionally create a wandb run
    # run = wandb.init(entity="..", project=f"Beras", name=f"Beras Test")
    run = None

    # 1. Create a SequentialModel using get_model

    # 2. Compile the model with optimizer, loss function, and accuracy metric
    
    # 3. Load and preprocess the data
    
    # 4. Train the model

    # 5. Evaluate the model

    # 6. save the predictions using np.save

    # 7. Call visualize_predictions

    # 8. Close the wandb run if you made one
    if run is not None: run.finish()