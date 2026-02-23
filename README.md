[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/_uYrfttt)
# CS1470 Deep Learning - Spring 2026

In this assignment, you will implement **BERAS**, a minimal neural network framework built on NumPy. You will build core components from scratch including layers, activations, loss functions, optimizers, and a training loop, then use your framework to classify handwritten digits from the MNIST dataset.

## Setup

### Environment

Make sure you have your CS1470 virtual environment activated. If you haven't set it up yet, refer to HW1.

## Assignment Structure

```
.
├── assignment.py        # Main file: build and train your model here
├── preprocess.py        # Data loading and preprocessing
├── visualize.py         # Visualization utilities
├── download.sh          # Data download script
├── test_runner.py       # Testing framework
├── beras/
│   ├── __init__.py      # Package exports
│   ├── core.py          # Tensor, Diffable, Callable base classes
│   ├── layers.py        # Linear (Dense) layer
│   ├── activations.py   # ReLU, LeakyReLU, Sigmoid, Softmax
│   ├── losses.py        # MSE, Categorical Cross-Entropy
│   ├── metrics.py       # Categorical Accuracy
│   ├── model.py         # Model and SequentialModel
│   ├── onehot.py        # One-Hot Encoder
│   └── optimizers.py    # SGD, RMSProp, Adam
└── tests/
    ├── test_utils.py    # Test runner utilities (DO NOT EDIT)
    ├── test_beras.py    # Tests for layers, activations, losses
    ├── test_data.py     # Tests for data preprocessing
    ├── test_gradient.py # Tests for gradient computation
    └── test_assignment.py # Tests for the full model
```

## Running Tests

Run all tests:
```bash
python test_runner.py
```

Run a specific test category:
```bash
python test_runner.py --category beras
python test_runner.py --category data
python test_runner.py --category gradient
python test_runner.py --category assignment
```

Run a specific test:
```bash
python test_runner.py --test test_dense_forward
```

List all available tests:
```bash
python test_runner.py --list
```

## Running the Assignment

```bash
python assignment.py
```
