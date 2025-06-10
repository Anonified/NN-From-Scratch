NN-From-Scratch
A minimalistic implementation of a neural network built entirely from scratch using Python, NumPy, and Matplotlib. This project demonstrates the fundamentals of neural networks, including forward propagation, backpropagation, and training, without relying on any deep learning frameworks.

🧠 Overview
This repository contains a simple feedforward neural network designed to classify handwritten digits from the MNIST dataset. The network architecture consists of:

784 input neurons (representing 28x28 pixel images)

2 hidden layers with 128 and 64 neurons, respectively

10 output neurons (one for each digit 0–9)

The network is trained using stochastic gradient descent (SGD) and employs the sigmoid activation function.

📈 Performance
Achieved an impressive 99.57% accuracy on the MNIST test set, demonstrating the effectiveness of implementing neural networks from scratch.

⚙️ Features
Implemented core neural network components: forward pass, backpropagation, and weight updates

Utilized NumPy for efficient numerical computations

No external machine learning libraries required

Clear and concise code suitable for educational purposes

📦 Dataset
The MNIST dataset is provided in the mnist.npz file included in this repository. This file contains the following arrays:
gist.github.com

training_images (shape: (60000, 784))

training_labels (shape: (60000, 10))

test_images (shape: (10000, 784))

test_labels (shape: (10000, 10))
github.com
+1
github.com
+1

Each image is flattened into a 784-dimensional vector, and labels are one-hot encoded.

🚀 Getting Started
Prerequisites
Ensure you have Python 3.x installed along with the following libraries:

NumPy

Matplotlib

Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/Anonified/NN-From-Scratch.git
cd NN-From-Scratch
Install the required libraries:

bash
Copy
Edit
pip install numpy matplotlib

Running the Script
To run the neural network training script, execute the following command in your terminal:

bash
Copy
Edit
python nn.py
Ensure you are in the directory containing nn.py and the mnist.npz file. The script will train the neural network and output the training progress and final accuracy.

