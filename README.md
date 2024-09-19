# Mini AutoGrad

This project is a minimalistic autograd engine inspired by and based on Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) repository. It implements a simple neural network library with automatic differentiation.

## Features

- Automatic differentiation
- Basic neural network operations
- Visualization of computation graphs

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/micrograd.git
   cd micrograd
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

You can use the `Value` class from `autograd.engine` to create computational graphs and perform automatic differentiation.

## Testing

To run the unit tests, follow these steps:

1. Ensure you're in the project root directory.
2. Run the following command:
   ```bash
   python -m unittest discover test
   ```

This command will discover and run all test files in the `test` directory.

## TODO

Future improvements planned for this project:

1. Implement tensor operations instead of scalar operations
2. Add support for convolutional layers
3. Implement additional activation functions:
    - Leaky ReLU
    - Sigmoid
    - Softmax
4. Add more optimization algorithms (e.g., Adam, RMSprop)
5. Improve performance with vectorized operations
6. Implement batch normalization
7. Add support for recurrent neural networks (RNNs)
8. Expand loss function support:
    - Mean Squared Error (MSE)
    - Binary Cross-Entropy
    - Categorical Cross-Entropy
