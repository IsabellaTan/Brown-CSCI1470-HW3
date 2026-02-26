#!/usr/bin/env python3
try:
    from .test_utils import setup_module_path, handle_cli_args
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tests.test_utils import setup_module_path, handle_cli_args
setup_module_path()
from preprocess import load_and_preprocess_data
from beras.core import Tensor
from beras.onehot import OneHotEncoder
import random
import numpy as np

"""
TESTING FILE FOR DATA.PY

The tests in this file are for your data preprocessing and utility functions! 
However, these are the minimal set of tests to ensure your code is correct.

Note: All test functions should start with 'test_' to be automatically discovered
by our testing framework.
"""

def test_preprocess_shapes():
    """Test the shape of the preprocessed data"""
    X_train, Y_train, X_test, Y_test = load_and_preprocess_data()

    # Check the shape of the preprocessed data
    assert X_train.shape == (60000, 784)
    assert X_test.shape == (10000, 784)
    assert Y_train.shape == (60000,)
    assert Y_test.shape == (10000,)
    print("Preprocess shapes test passed!")

def test_preprocess_values():
    """Test the values of the preprocessed data"""
    X_train, _, X_test, _ = load_and_preprocess_data()

    # Check the values of the preprocessed data
    are_all_between_0_and_1 = ((X_train >= 0) & (X_train <= 1)).all()
    assert are_all_between_0_and_1, "X_train is not between 0 and 1"
    are_all_between_0_and_1 = ((X_test >= 0) & (X_test <= 1)).all()
    assert are_all_between_0_and_1, "X_test is not between 0 and 1"

    print("Preprocess values test passed!")

def test_ohe_validity():
    """Test the validity of the OneHotEncoder"""

    # Generate random data
    length = random.randint(2, 500)
    data = np.random.randint(50, size=length)
    stud_one_hot = OneHotEncoder()

    # Fit the OneHotEncoder
    stud_one_hot.fit(data)

    encoded = stud_one_hot.forward(data)        
    
    # check shape
    num_classes = len(np.unique(data))
    assert encoded.shape == (length, num_classes)
    
    row_sums = encoded.sum(axis=1)
    assert np.all(row_sums == 1)
    assert np.all(np.isin(encoded, [0, 1]))

    print("OHE encoding is valid!")


# ============================================================================
# TODO: Add more tests to ensure your code is correct!
# ============================================================================
def test_preprocess_dtype():
    """Test that preprocessed inputs are float type"""
    X_train, Y_train, X_test, Y_test = load_and_preprocess_data()

    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)

    assert np.issubdtype(X_train.dtype, np.floating)
    assert np.issubdtype(X_test.dtype, np.floating)

    print("Preprocess dtype test passed!")

def test_label_range():
    """Test that labels are between 0 and 9"""
    _, Y_train, _, Y_test = load_and_preprocess_data()

    assert np.all((Y_train >= 0) & (Y_train <= 9))
    assert np.all((Y_test >= 0) & (Y_test <= 9))

    print("Label range test passed!")

def test_no_nan_values():
    """Ensure no NaN values in dataset"""
    X_train, Y_train, X_test, Y_test = load_and_preprocess_data()

    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
    assert not np.isnan(Y_train).any()
    assert not np.isnan(Y_test).any()

    print("No NaN values test passed!")

def test_flattened_input():
    """Ensure images are flattened to 784 features"""
    X_train, _, X_test, _ = load_and_preprocess_data()

    assert X_train.shape[1] == 784
    assert X_test.shape[1] == 784

    print("Flatten test passed!")

def test_ohe_deterministic():
    """Ensure OHE produces consistent encoding"""
    data = np.array([1, 2, 3, 2, 1])
    encoder = OneHotEncoder()
    encoder.fit(data)

    encoded1 = encoder.forward(data)
    encoded2 = encoder.forward(data)

    assert np.array_equal(encoded1, encoded2)

    print("OHE deterministic test passed!")


# ============================================================================
# TEST RUNNERS! DO NOT EDIT THIS SECTION!
# ============================================================================
try:
    from .test_utils import run_single_test, list_available_tests, run_all_tests
except ImportError:
    from tests.test_utils import run_single_test, list_available_tests, run_all_tests

# Convenience functions that wrap the utilities with this module's context
def run_test(test_name: str):
    """Run a single test by name. Usage: run_test('test_preprocess_shapes')"""
    return run_single_test(test_name, globals(), 'test_data')

def list_tests():
    """List all available tests in this module."""
    return list_available_tests(globals(), 'test_data')

def run_all():
    """Run all tests in this module."""
    return run_all_tests(globals(), 'test_data')

if __name__ == '__main__':
    handle_cli_args(globals(), 'test_data')
