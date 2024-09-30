import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load and preprocess the MNIST dataset
def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize pixel values to between 0 and 1
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Add a channel dimension (required for CNNs)
    X_train = X_train[..., tf.newaxis]
    X_test = X_test[..., tf.newaxis]

    return X_train, y_train, X_test, y_test
