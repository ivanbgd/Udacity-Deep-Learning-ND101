# Prediction accuracy: 0.725 Nice job! That's right!

# For loops are too slow compared to Numpy!!!

import numpy as np
from data_prep import features, targets, features_test, targets_test

# Activation function
def sigmoid(x):
    return 1./(1. + np.exp(-x))

# Neural Network hyperparameters
n_epochs = 1000
learn_rate = .5     # Eta

# Features are inputs to our NN, and there's 6 of them.
# We have 360 train data points (records).
n_points, n_features = features.shape               # shape (360, 6)

# Initial weights - They should be small and random, around 0, so that inputs are in the linear region of the sigmoid.
weights = np.random.normal(loc = 0.0, scale = 1./np.sqrt(n_features), size = n_features)

last_loss = None

for epoch in range(n_epochs):
    delta_w = np.zeros(weights.shape)               # shape (6,)
    output = sigmoid(np.dot(features, weights))     # shape (360,)
    error = targets - output                        # shape (360,)
    error_term = error * output * (1 - output)      # shape (360,)
    delta_w += np.dot(error_term, features)         # shape (6,) - The gradient descent step, the error times the gradient times the inputs
    mse = .5 * np.mean(np.square(error))            # shape (), that is, a scalar
    weights += learn_rate * delta_w / n_points      # shape (6,)

    # Printing out the mean square error on the training set
    if epoch % (n_epochs / 10) == 0:
        loss = mse
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss


# Calculate accuracy on test data
test_out = sigmoid(np.dot(features_test, weights))
predictions = test_out > 0.5    # When sigmoid > 0.5, that is logical "1"; below 0.5 is logical "0".
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
