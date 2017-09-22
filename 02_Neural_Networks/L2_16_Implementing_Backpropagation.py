import numpy as np
from data_prep import features, targets, features_test, targets_test

np.random.seed(21)

# Activation function
def sigmoid(x):
    return 1./(1. + np.exp(-x))

# Neural Network hyperparameters
n_hidden = 2        # number of hidden units
n_epochs = 900
learn_rate = .005   # Eta

# Features are inputs to our NN, and there's 6 of them.
# We have 360 train data points (records).
n_records, n_features = features.shape                                                                          # shape (360, 6)

# Initial weights - They should be small and random, around 0, so that inputs are in the linear region of the sigmoid.
weights_input_hidden = np.random.normal(loc = 0.0, scale = n_features**-.5, size = (n_features, n_hidden))      # shape (6, 2)
weights_hidden_output = np.random.normal(loc = 0.0, scale = n_features**-.5, size = (n_hidden,))                # shape (2,) - But, I think it should be n_hidden**-.5

last_loss = None

for epoch in range(n_epochs):
    hidden_input = np.dot(features, weights_input_hidden)                                                       # shape (360, 2)
    hidden_output = sigmoid(hidden_input)                                                                       # shape (360, 2)
    output_input = np.dot(hidden_output, weights_hidden_output)                                                 # shape (360,)
    output_output = sigmoid(output_input)                                                                       # shape (360,)
    error = targets - output_output                                                                             # shape (360,)
    output_error_term = error * output_output*(1 - output_output)                                               # shape (360,)
    hidden_error_term = np.dot(output_error_term[:, np.newaxis], weights_hidden_output[np.newaxis, :]) * hidden_output*(1 - hidden_output)    # shape (360, 2)
    delta_w_i_h = np.dot(features.T, hidden_error_term)                                                        # shape (6, 2) - The gradient descent step, the error times the gradient times the inputs
    delta_w_h_o = np.dot(output_error_term, hidden_output)                                                     # shape (2,) - The gradient descent step, the error times the gradient times the inputs
    weights_input_hidden += learn_rate * delta_w_i_h / n_records
    weights_hidden_output += learn_rate * delta_w_h_o / n_records
    mse = .5 * np.mean(np.square(error))                                                                        # shape (), that is, a scalar

    # Printing out the mean square error on the training set
    if epoch % (n_epochs / 10) == 0:
        loss = mse
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden_out = sigmoid(np.dot(features_test, weights_input_hidden))
output_out = sigmoid(np.dot(hidden_out, weights_hidden_output))
predictions = output_out > 0.5      # When sigmoid > 0.5, that is logical "1"; below 0.5 is logical "0".
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))

# Prediction accuracy: 0.725 Nice job! That's right!
# Neural Network hyperparameters 3, 2000 and 1.005 give accuracy of 0.750.


#print()
#print('weights_input_hidden: ')
#print(weights_input_hidden)
#print('weights_hidden_output: ')
#print(weights_hidden_output)

