#!/usr/bin/env python3
try:
    from .test_utils import setup_module_path, handle_cli_args
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tests.test_utils import setup_module_path, handle_cli_args
setup_module_path()
import numpy as np
import beras
import random
from beras import Tensor, Linear, LeakyReLU, MeanSquaredError
import torch
from torch import nn
from torch.nn import functional as F
import tensorflow as tf


"""
TESTING FILE FOR BERAS.PY

The tests in this file are for your layers, losses, activations, and metrics! 
However, these are the minimal set of tests to ensure your code is correct.

In fact, you are responsible for implementing the basic tests for sigmoid,
softmax, and CCE loss! You should add more tests to ensure your code is correct.

Note: All test functions should start with 'test_' to be automatically discovered
by our testing framework.
"""

def test_dense_forward():
    """Test the forward pass of your Dense layer"""

    # Instantiate your Dense layer and Keras Dense layer
    your_dense = Linear(10, 5)
    torch_linear = nn.Linear(10, 5)

    # Generate a random test input
    test_input = np.random.randn(10, 10)

    # Compute the forward pass of your Dense layer and Keras Dense layer
    your_out = your_dense(test_input)
    torch_out = torch_linear(torch.tensor(test_input).float())
    
    # Compare the shapes of the output
    assert torch_out.shape == your_out.shape, f"Shapes differ: {torch_out.shape} vs {your_out.shape}"
    print("Forward pass successful")

def test_leaky_relu():
    """
    Test the forward pass of the LeakyReLU activation function.
    """

    # Instantiate your LeakyReLU activation function and Keras LeakyReLU activation function
    student_leaky_relu = beras.LeakyReLU()
    leaky_relu = nn.LeakyReLU(negative_slope=0.3)

    # Generate a random test array
    test_arr = np.array(np.arange(-8,8),np.float64)

    # Compare the forward pass of your LeakyReLU activation function and Keras LeakyReLU activation function
    assert(all(np.isclose(student_leaky_relu(test_arr),leaky_relu(torch.tensor(test_arr)))))
    print("Leaky ReLU test passed!")


def test_sigmoid():
    """Use the Leaky ReLU test as a guide to test your Sigmoid activation!"""

    # TODO: Implement this test
    student_sigmoid = beras.activations.Sigmoid()
    sigmoid = nn.Sigmoid()
    test_arr = np.array(np.arange(-8,8),np.float32)
    assert(all(np.isclose(student_sigmoid(test_arr),sigmoid(torch.tensor(test_arr).float()))))
    print("Sigmoid test passed!")

def test_softmax():
    """Use the Leaky ReLU test as a guide to test your Softmax activation!"""

    # TODO: Implement this test
    student_softmax = beras.activations.Softmax()

    test_arr = np.random.rand(3, 4)

    assert(np.all(np.isclose(student_softmax(test_arr), F.softmax(torch.tensor(test_arr).float(), dim=-1))))
    print("Softmax test passed!")

def test_mse_forward():
    """
    Test the forward pass of the MeanSquaredError loss function. 
    """

    # Instantiate your MeanSquaredError loss function
    beras_mse = beras.MeanSquaredError()

    # Generate random test cases
    x = np.random.randint(0, 10, size=(2, 3))
    y = np.random.randint(0, 10, size=(2, 3))

    solution_mse = nn.MSELoss()(torch.tensor(x).float(), torch.tensor(y).float())

    # Compare the solution MSE and your MSE
    assert np.allclose(solution_mse, beras_mse(x, y))

    print("MSE test passed!")

def test_cce():
    """
    Test the forward pass of the CategoricalCrossentropy loss function.
    """
    # Generate random test cases
    batch_size, num_classes = 5, 4

    true_labels = np.random.randint(0, num_classes, batch_size)
    y_true = np.eye(num_classes)[true_labels]
    
    # Random predictions (softmax normalized)
    logits = np.random.randn(batch_size, num_classes)
    y_pred = beras.activations.Softmax()(logits)
    
    # Our implementation
    our_loss = beras.losses.CategoricalCrossEntropy()(y_pred, y_true)

    torch_cce = nn.CrossEntropyLoss()
    
    # Torch implementation. Notice that torch expects unsoft-maxed logits -> fusing the ops is faster
    torch_loss = torch_cce(torch.tensor(logits), torch.tensor(true_labels)).numpy()
    
    # Should be very close
    np.testing.assert_allclose(our_loss, torch_loss)
    print("CCE test passed")

def test_categorical_accuracy():
    y_true = [[0, 0, 1], [0, 1, 0]]
    y_pred = np.random.uniform(0, 1, size=(2, 3))
    student_acc = beras.metrics.CategoricalAccuracy()(y_pred,y_true)
    acc = tf.keras.metrics.categorical_accuracy(y_true,y_pred)
    assert(student_acc == np.mean(acc))
    print("Categorical accuracy test passed")

# # ============================================================================
# # TODO: Add more tests to ensure your code is correct!
# # ============================================================================
def test_softmax_row_sums():
    student_softmax = beras.activations.Softmax()

    test_arr = np.random.randn(5, 10)
    out = student_softmax(test_arr)

    row_sums = np.sum(out, axis=1)

    np.testing.assert_allclose(row_sums, np.ones(5), atol=1e-6)

    print("Softmax row sum test passed!")

def test_dense_output_shape():
    layer = Linear(7, 3)

    x = np.random.randn(11, 7)
    out = layer(x)

    assert out.shape == (11, 3)

    print("Dense output shape test passed!")

def test_sigmoid_extreme_values():
    student_sigmoid = beras.activations.Sigmoid()

    test_arr = np.array([-1000, -100, 0, 100, 1000], dtype=np.float32)

    out = student_sigmoid(test_arr)

    assert np.all(out >= 0)
    assert np.all(out <= 1)

    print("Sigmoid extreme value test passed!")

# ============================================================================
# TEST RUNNERS! DO NOT EDIT THIS SECTION!
# ============================================================================
try:
    from .test_utils import run_single_test, list_available_tests, run_all_tests
except ImportError:
    from tests.test_utils import run_single_test, list_available_tests, run_all_tests

# Convenience functions that wrap the utilities with this module's context
def run_test(test_name: str):
    """Run a single test by name. Usage: run_test('test_dense_forward')"""
    return run_single_test(test_name, globals(), 'test_beras')

def list_tests():
    """List all available tests in this module."""
    return list_available_tests(globals(), 'test_beras')

def run_all():
    """Run all tests in this module."""
    return run_all_tests(globals(), 'test_beras')

if __name__ == "__main__":
    handle_cli_args(globals(), 'test_beras')