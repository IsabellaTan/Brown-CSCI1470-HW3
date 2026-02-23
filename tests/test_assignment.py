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
import torch
from torch import nn

import assignment
import beras

"""
THESE ARE TESTS FOR ASSIGNMENT.PY

This is a minimal set of tests for the assignment file.
You should add more robust tests to ensure you code is correct!
Note:  All test functions should start with 'test_' to be discovered
by our testing framework.
"""

def test_forward():
    """Test forward pass of your model compared to Keras model"""

    # Instantiate your model components
    model = beras.Sequential(
        beras.Linear(10, 5, initializer="kaiming", name="Dense1"),
        beras.ReLU(),
        beras.Linear(5, 1, initializer="kaiming", name="Dense2"),
    )

    # Instantiate the Keras model
    torch_model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )

	# Generate a random test input
    x = np.random.uniform(0, 1, size=(1, 10))

    # Compute the forward pass of your model and Keras model
    student_output = model(x)
    torch_output = torch_model(torch.tensor(x).float())

	# Compare the shapes of the output
    assert student_output.shape == torch_output.shape, f"Output shapes differ: {student_output.shape} vs {torch_output.shape}"
    print("Model forward pass shapes match!")

def test_model_return():
	"""Test the return shape of the entire compiled model"""
	model = assignment.get_model()

	fake_data = np.zeros((353, 784))

	assert model(fake_data).shape == (353, 10)
	print("Model return shape test passed!")

# ============================================================================
# TODO: Add more tests to ensure your code is correct!
# ============================================================================


# ============================================================================
# TEST RUNNERS! DO NOT EDIT THIS SECTION!
# ============================================================================
try:
    from .test_utils import run_single_test, list_available_tests, run_all_tests
except ImportError:
    from tests.test_utils import run_single_test, list_available_tests, run_all_tests

# Convenience functions that wrap the utilities with this module's context
def run_test(test_name: str):
    """Run a single test by name. Usage: run_test('test_forward')"""
    return run_single_test(test_name, globals(), 'test_assignment')

def list_tests():
    """List all available tests in this module."""
    return list_available_tests(globals(), 'test_assignment')

def run_all():
    """Run all tests in this module."""
    return run_all_tests(globals(), 'test_assignment')

if __name__ == "__main__":
    handle_cli_args(globals(), 'test_assignment')
