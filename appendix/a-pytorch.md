# PyTorch

PyTorch's three main components include:
1. a tensor library as a fundamental building block for computing,
  * Extends Numpy arrays with GPU acceleration.
2. automatic differentiation for model optimization,
  * Also known as "autograd".
  * Automatic computation of gradients for tensor operations.
  * Simplifying back propagation and model optimization,
3. and deep learning utility functions.
  * Pre-trained models, loss functions, optimizers.

![PyTorch 3 main components](./A-1-pytorch-three-main-components.png)

## Tensors

[**Tensors**](https://en.wikipedia.org/wiki/Tensor) are a mathematical concept that generalizes vectors and matrices to potentially higher dimensions. In other words, tensors are mathematical objects that can be characterized by their order (or rank), which provides the number of dimensions. For example, a scalar (just a number) is a tensor of rank 0, a vector is a tensor of rank 1, and a matrix is a tensor of rank 2.

## Automatic differentiation made easy

PyTorch builds a computational graph internally by default if one of its terminal nodes has the `requires_grad` attribute set to `True`. This is useful if we want to compute gradients. Gradients are required when training neural networks via the popular backpropagation algorithm, which can be considered an implementation of the chain rule from calculus for neural networks, illustrated in figure A.8.

![chain rule for neural networks](./A-8-chain-rule-for-neural-networks.png)

Figure A.8 The most common way of computing the loss gradients in a computation graph involves applying the chain rule from right to left, also called reverse-model automatic differentiation or backpropagation. We start from the output layer (or the loss itself) and work backward through the network to the input layer. We do this to compute the gradient of the loss with respect to each parameter (weights and biases) in the network, which informs how we update these parameters during training.

Figure A.8 shows partial derivatives, which measure the rate at which a function changes with respect to one of its variables. A gradient is a vector containing all of the partial derivatives of a *multivariate function* (a function with more than one variable as input).

The chain rule is a way to compute gradients of a loss function given the model's parameters in a computation graph. This provides the information needed to update each parameter to minimize the loss function, which serves as a proxy for measuring the model's performance using a method such as gradient descent.

