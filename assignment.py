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
        Linear(784, 128, initializer="kaiming"),
        ReLU(),
        Linear(128, 64, initializer="kaiming"),
        ReLU(),
        Linear(64, 10, initializer="xavier"),
        Softmax()
    )
    return model

def get_optimizer(model: Model):
    # choose an optimizer, initialize it and return it!
    return Adam(model.parameters, learning_rate=0.001)

def get_loss_fn():
    # choose a loss function, initialize it and return it!
    return CategoricalCrossEntropy()

def get_acc_fn():
    # choose an accuracy metric, initialize it and return it!
    return CategoricalAccuracy()

if __name__ == '__main__':

    ### Use this area to test your implementation!
    Tensor.debug_print = False #turn to True for extra print statements during compose!

    # 0. Optionally create a wandb run
    # run = wandb.init(entity="..", project=f"Beras", name=f"Beras Test")
    run = None

    # 1. Create a SequentialModel using get_model
    model = get_model()
    # 2. Compile the model with optimizer, loss function, and accuracy metric
    optimizer = get_optimizer(model)
    loss_fn = get_loss_fn()
    acc_fn = get_acc_fn()
    model.compile(optimizer, loss_fn, acc_fn)
    # 3. Load and preprocess the data
    train_inputs, train_labels, test_inputs, test_labels = load_and_preprocess_data()
    ohe = OneHotEncoder()
    train_labels = ohe.forward(train_labels)
    test_labels = ohe.forward(test_labels)
    # 4. Train the model
    model.fit(
        train_inputs,
        train_labels,
        epochs=5,
        batch_size=64,
        wandb_run=run
    )
    # 5. Evaluate the model
    metrics, predictions = model.evaluate(
        test_inputs,
        test_labels,
        batch_size=64,
        wandb_run=run
    )
    print("Test Metrics:", metrics)
    # 6. save the predictions using np.save
    np.save("predictions.npy", predictions)
    # 7. Call visualize_predictions
    visualize_predictions(model, test_inputs, test_labels)
    # 8. Close the wandb run if you made one
    if run is not None: run.finish()



