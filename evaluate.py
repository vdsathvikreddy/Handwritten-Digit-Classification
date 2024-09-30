from tensorflow.keras.models import load_model
from load_data import load_mnist

# Evaluate the model on the test data
def evaluate(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc * 100:.2f}%")

# Load data
X_train, y_train, X_test, y_test = load_mnist()

# Evaluate the ANN
evaluate(ann_model, X_test, y_test)

# Evaluate the CNN
evaluate(cnn_model, X_test, y_test)
