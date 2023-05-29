# EasyGrad


## Overview

This project is a modified version of the Micrograd library, which provides a tiny autograd engine for reverse-mode automatic differentiation of scalar values. Specifically, I have added some loss function on the Value class, and have modified the implementation of the MLP to include an option to indicate the activation during initialization.

The purpose of this project is to serve as an educational resource and demonstrate the principles of reverse-mode autodiff and neural networks in a concise and accessible manner. The modified version includes updates and improvements specific to my personal use case.

## Usage

#### Create Value objects
a = Value(-4.0)

b = Value(2.0)

#### Perform operations
c = a + b

d = a * b + b**3

c += c + 1

c += 1 + c + (-a)

d += d * 2 + (b + a).relu()

d += 3 * d + (b - a).relu()

e = c - d

f = e**2

#### Create a neuron
neuron = Neuron(n_inputs=3, activation='relu')

#### Perform forward pass
output = neuron([1, 2, 3])

#### Create a layer
layer = Layer(nin=3, nout=2, activation='sigmoid')

#### Perform forward pass
output = layer([1, 2, 3])

#### Create a Multi-Layer Perceptron
model = MLP(10, [(5, 'relu'), (5, 'relu'), (1, 'sigmoid')])

#### Perform forward pass
output = model([1, 2, 3])

## Acknowledgements
This library is based on the original micrograd library developed by [karpathy](https://github.com/karpathy/micrograd)
