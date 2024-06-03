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
```
This will:

- Load and preprocess the MNIST dataset.
- Build and compile the CNN model.
- Train the model on the training data.
- Evaluate the model on the test data.
- Save the trained model to a file named mnist_cnn_model.h5.

## Testing custom images

You can test the trained model on your own handwritten digit images by following these steps:

1. Prepare Custom Images

Create your own handwritten digit images and save them in the my_digits/ directory. Ensure the images are on a plain white background and saved as PNG or JPEG files.
Run the Prediction Script

Use the script included in the mnist_cnn.py file to preprocess your images and make predictions:
```
python mnist_cnn.py
```
The script will preprocess your images, use the trained model to make predictions, and print the predicted digits.

## Contributing
Contributions are welcome! If you have any improvements or suggestions, please create a pull request or open an issue.

## License
This project is licensed under the MIT License. See the LICENSE file for details.


### Instructions to Push the Project to GitHub

1. **Initialize a Git Repository**
    ```bash
    git init
    ```

2. **Add Files to the Repository**
    ```bash
    git add .
    ```

3. **Commit the Files**
    ```bash
    git commit -m "Initial commit"
    ```

4. **Create a New Repository on GitHub**
    - Go to GitHub and create a new repository (e.g., `handwritten-digit-recognition`).

5. **Add the Remote Repository**
    ```bash
    git remote add origin https://github.com/your-username/handwritten-digit-recognition.git
    ```

6. **Push the Files to GitHub**
    ```bash
    git push -u origin master
    ```
