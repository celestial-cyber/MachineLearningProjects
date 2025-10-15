"""
Image Classification on MNIST Dataset using a Convolutional Neural Network (CNN) 
with Fully Connected layers (Dense layers).

This program trains a CNN on the MNIST dataset, which contains 28x28 grayscale images 
of handwritten digits (0-9). The model will learn to classify the images into the 
correct digit class.
"""

# --------------------------- Import Libraries ---------------------------
import matplotlib.pyplot as plt                # For plotting graphs (accuracy vs epochs)
import tensorflow as tf                       # Core TensorFlow library for building ML models
from tensorflow.keras import datasets, layers, models  
# datasets -> for loading MNIST
# layers -> for building CNN layers (Conv2D, MaxPooling2D, Dense, Flatten)
# models -> to create a Sequential model

# --------------------------- Load Dataset ---------------------------
# MNIST dataset comes pre-split into training and testing sets
# x_train, x_test -> image data
# y_train, y_test -> labels (digits 0-9)
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# --------------------------- Normalize Data ---------------------------
# Pixel values of images range from 0 to 255
# Dividing by 255 scales values to [0,1], which helps neural networks train faster and more accurately
x_train, x_test = x_train / 255.0, x_test / 255.0

# --------------------------- Reshape Data ---------------------------
# CNNs expect input in 4D: (num_samples, height, width, channels)
# MNIST images are grayscale, so channels = 1
# '-1' means automatically calculate the number of samples
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Print shape of training dataset for verification
print("Training Dataset shape:", x_train.shape)
# Example output: (60000, 28, 28, 1)
# Explanation:
# 60000 -> number of training images
# 28x28 -> height and width of each image
# 1 -> number of channels (grayscale)

# --------------------------- Build CNN Model ---------------------------
# Sequential model: stack layers one by one
model = models.Sequential([
    
    # First convolutional layer
    layers.Conv2D(
        filters=32,           # Number of feature maps / filters
        kernel_size=(3,3),    # Size of each filter (3x3 pixels)
        activation='relu',    # ReLU activation adds non-linearity
        input_shape=(28,28,1) # Input shape for first layer
    ),

    # Second convolutional layer
    layers.Conv2D(
        filters=64,           # More filters to learn more complex features
        kernel_size=(3,3),
        activation='relu'
    ),

    # Max Pooling layer
    layers.MaxPooling2D(
        pool_size=(2,2)       # Downsamples the feature maps by taking max of each 2x2 block
    ),

    # Flatten layer
    # Converts 2D feature maps into 1D vector for the fully connected (Dense) layer
    layers.Flatten(),

    # Fully connected layer
    layers.Dense(
        64,                   # Number of neurons
        activation='relu'     # ReLU activation
    ),

    # Output layer
    layers.Dense(
        10,                   # Number of classes (digits 0-9)
        activation='softmax'  # Softmax outputs probability distribution over classes
    )
])

# --------------------------- Compile Model ---------------------------
# Before training, we must compile the model by specifying:
# - Optimizer: 'adam' (adaptive optimizer, adjusts learning rate automatically)
# - Loss function: 'sparse_categorical_crossentropy' (for integer labels)
# - Metrics: ['accuracy'] to track training performance
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --------------------------- Train the Model ---------------------------
# Fit the model on training data
# epochs=5 -> number of complete passes through the training dataset
# validation_data -> evaluate model performance on test set during training
history = model.fit(
    x_train, y_train,
    epochs=5,
    validation_data=(x_test, y_test)
)

# --------------------------- Evaluate the Model ---------------------------
# Evaluate performance on test dataset
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("\nTest Accuracy:", test_acc)
# Expected output example: 0.987
# Meaning: The CNN correctly classifies ~98.7% of the test images

# --------------------------- Plot Accuracy ---------------------------
# Plot training vs validation accuracy over epochs
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy on MNIST')
plt.legend()
plt.show()
# Interpretation:
# - If train and validation curves are close → model generalizes well
# - If train >> validation → model overfitting
"""✅ Detailed Explanation of Each Step

Importing Libraries – Necessary for model building, dataset loading, and plotting results.

Loading Dataset – MNIST is pre-split into training and test sets for supervised learning.

Normalizing Data – Neural networks train faster and avoid exploding gradients when inputs are scaled between 0 and 1.

Reshaping Data – CNN expects 4D input: (num_samples, height, width, channels).

Building the CNN

Conv2D layers extract features (edges, shapes, patterns).

MaxPooling2D reduces spatial dimensions and computation.

Flatten prepares data for fully connected layers.

Dense layers learn complex patterns and make final classification.

Compiling the Model – Defines how the model is optimized and evaluated.

Training (fit) – Model learns patterns from data over multiple epochs; validation data monitors generalization.

Evaluation – Measures accuracy on unseen test data.

Plotting Accuracy – Visualizes learning performance, helps detect overfitting/underfitting."""
