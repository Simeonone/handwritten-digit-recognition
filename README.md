# Handwritten Digit Recognition with CNN

This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits using the MNIST dataset. The model is trained using TensorFlow and Keras, and it can be tested on custom images of handwritten digits.

## Table of Contents
- [Overview](#overview)
- [Setup Instructions](#setup-instructions)
- [Training the Model](#training-the-model)
- [Testing Custom Images](#testing-custom-images)
- [Contributing](#contributing)
- [License](#license)

## Overview

The goal of this project is to build a CNN that can accurately classify images of handwritten digits (0-9). The MNIST dataset, which contains 60,000 training images and 10,000 test images, is used to train and evaluate the model.

## Setup Instructions

1. **Clone the Repository**

    ```bash
    git clone https://github.com/Simeonone/handwritten-digit-recognition.git
    cd handwritten-digit-recognition
    ```

2. **Create a Virtual Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**

    ```bash
    pip install tensorflow
    pip install numpy
    pip install matplotlib
    pip install opencv-python
    ```

## Training the Model

The training script is included in the `mnist_cnn.py` file. To train the model, simply run the script:

```bash
python mnist_cnn.py
