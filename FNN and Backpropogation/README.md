# Feedforward Neural Network from Scratch

This folder contains a simple Multi-Layer Feedforward Neural Network (MLFFNN) built completely from scratch using just Python and NumPy. It's a great way to understand the math behind neural networks without relying on heavy frameworks like PyTorch or TensorFlow.

## What's inside?

- `Implementation of MLFNN and Backpropogation.py`: A Python script that builds out the neural network class `MLFFNN`. 

The network includes everything you need to see how it learns under the hood:
- Forward pass logic
- ReLU and Softmax activations (along with their derivatives)
- Categorical cross-entropy loss calculation
- Full backpropagation (computing gradients using the chain rule)
- Basic gradient descent to update weights and biases

At the bottom of the script, there's a quick example where the network is trained on the classic XOR problem. We set up an architecture of `[2, 8, 2]` (2 inputs, 8 hidden neurons, 2 outputs) to show that it actually learns non-linear data!

## How to run it

Just make sure you have `numpy` installed, and then run the Python script:

```bash
pip install numpy
python "Implementation of MLFNN and Backpropogation.py"
```

The script will train for 5000 epochs and print the loss going down every 500 epochs. At the end, you'll see its final predictions compared to the actual XOR labels.
