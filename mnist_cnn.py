import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images to the range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape the data to include the channel dimension (since the images are grayscale)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert the labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build the model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Callbacks
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(factor=0.2, patience=3)

# Train the model
model.fit(datagen.flow(x_train, y_train, batch_size=32),
          validation_data=(x_test, y_test),
          epochs=20,
          steps_per_epoch=len(x_train) // 32,
          callbacks=[early_stopping, lr_reduction])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy}')

# Predict the first 10 images from the test set
predictions = model.predict(x_test[:10])

# Plot the images and their predicted labels
for i in range(10):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f'Predicted Label: {np.argmax(predictions[i])}')
    plt.show()

# User-input image processing - uncomment to run for custom user images
'''
def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize the image to 28x28
    img = cv2.resize(img, (28, 28))
    
    # Normalize the image
    img = img.astype('float32') / 255.0
    
    # Reshape the image to match the input shape of the model
    img = img.reshape(1, 28, 28, 1)
    
    return img

def classify_image(model, image_path):
    # Preprocess the image
    img = preprocess_image(image_path)
    
    # Make a prediction
    prediction = model.predict(img)
    
    # Get the predicted class
    predicted_class = np.argmax(prediction)
    
    return predicted_class

# Example usage
user_image_path = 'path/to/user/image.jpg'
predicted_class = classify_image(model, user_image_path)
print(f"The predicted class for the user's image is: {predicted_class}")

# Display the user's image
user_img = cv2.imread(user_image_path, cv2.IMREAD_GRAYSCALE)
plt.imshow(user_img, cmap='gray')
plt.title(f'Predicted Label: {predicted_class}')
plt.show()
'''