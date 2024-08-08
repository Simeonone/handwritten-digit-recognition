# MNIST Digit Recognition with TensorFlow

This project implements a Convolutional Neural Network (CNN) for recognizing handwritten digits using the MNIST dataset. It also includes functionality for classifying user-input images.

## Features

- CNN model trained on the MNIST dataset
- Data augmentation to improve model generalization
- Early stopping and learning rate reduction callbacks
- Ability to process and classify user-input images

## Requirements

- Python 3.8+
- TensorFlow 2.17.0
- NumPy
- Matplotlib
- OpenCV

For a complete list of dependencies, see `requirements.txt`.

## Installation

1. Clone this repository:
   
`git clone https://github.com/Simeonone/handwritten-digit-recognition.git`

`cd handwritten-digit-recognition`

2. Create a virtual environment (optional but recommended):
  
`python -m venv venv`

`source venv/bin/activate`  On Windows, use `venv\Scripts\activate`

3. Install the required packages:

`pip install -r requirements.txt`

## Usage

1. Train the model and evaluate on MNIST dataset:

`python mnist_cnn.py`

2. To classify your own image:
- Uncomment the user-input image processing section at the end of `mnist_cnn.py`
- Replace `'path/to/user/image.jpg'` with the path to your image
- Run the script again

## Model Architecture

The CNN model consists of:
- 2 Convolutional layers with ReLU activation
- 2 MaxPooling layers
- Flatten layer
- Dense layer with ReLU activation and Dropout
- Output Dense layer with Softmax activation

## Results

The model achieves 0.9940000176429749 accuracy on the MNIST test set.

## Contributing

Contributions, issues, and feature requests are welcome. 


## Acknowledgments

- MNIST dataset providers
- TensorFlow and Keras documentation