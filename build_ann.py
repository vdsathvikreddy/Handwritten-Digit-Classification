from tensorflow.keras import layers, models

# Build a simple ANN model using TensorFlow/Keras
def build_ann():
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(28 * 28,)),  # Input layer with 784 inputs (28x28)
        layers.Dense(64, activation='relu'),                           # Hidden layer with 64 neurons
        layers.Dense(10, activation='softmax')                        # Output layer with 10 neurons for classification
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
