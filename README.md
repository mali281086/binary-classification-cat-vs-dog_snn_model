# Neural Network from Scratch: Cats vs Dogs 🧠

This project demonstrates a basic **feedforward neural network** built entirely from scratch using **NumPy** to classify cat and dog images. It includes manual implementation of forward and backward propagation, gradient descent, and prediction — no external ML libraries involved.

## 📌 Project Overview

- Load and preprocess image datasets using Pillow and NumPy
- Build a two-layer neural network with:
  - Tanh activation in the hidden layer
  - Sigmoid activation in the output layer (for binary classification)
- Manually implement forward and backward propagation
- Train the model using cross-entropy loss and gradient descent
- Predict on new images and visualize the cost over iterations

## 🛠️ Features

- No TensorFlow, PyTorch, or scikit-learn — pure NumPy!
- Easily adjustable hidden layer size (`n_h`)
- Prediction function to classify your own images
- Image normalization, reshaping, and flattening
- Cost visualization to track model learning

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.x installed, then install the required packages:

```bash
pip install -r requirements.txt

Folder Structure
project-root/
├── Neural_Network_From_Scratch.ipynb   # Complete working notebook
├── requirements.txt                    # Dependencies list
├── README.md                           # GitHub project readme
├── info.txt                            # Project site info
└── dataset/
    ├── Cat/
    └── Dog/

⚠️ dataset/ folder should contain your preprocessed training/test images. You may need to manually prepare/rescale images to 64x64.

🧪 How to Use
Clone the repository and open the notebook in Jupyter.
Make sure the dataset is organized into Cat/ and Dog/ subfolders.
Run all cells to:
Train the neural network
Visualize learning progress
Predict on a new image (customizable path)
Adjust the number of hidden units or learning rate as needed.

📷 Custom Predictions
You can predict a new image using:

from PIL import Image
img = Image.open('./dataset/manual_set/4.jpg').resize((64, 64)).convert('RGB')
img_array = np.array(img).reshape(-1, 1) / 255.0
A2, _ = forward_propagation(img_array, parameters)
prediction = (A2 > 0.5)

📖 License
This project is for educational purposes. Feel free to fork, modify, and experiment!