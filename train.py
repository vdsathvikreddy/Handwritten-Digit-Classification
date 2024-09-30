from build_cnn import build_cnn
from build_ann import build_ann
from load_data import load_mnist

# Train the model
def train(model, X_train, y_train, num_epochs=10):
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=32, verbose=2)

# Load data
X_train, y_train, X_test, y_test = load_mnist()

# Train the ANN
ann_model = build_ann()
train(ann_model, X_train, y_train, num_epochs=10)

# Train the CNN
cnn_model = build_cnn()
train(cnn_model, X_train, y_train, num_epochs=10)
