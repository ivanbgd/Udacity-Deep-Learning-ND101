import tensorflow as tf
import os
from glob import glob


# Remove all files previously created in this script
try:
    os.remove('./checkpoint')
except FileNotFoundError:
    pass
for i in range(100):    # one iteration per file
    try:
        os.remove(glob('./*model.ckpt*.*', recursive=False)[0])     # deletes one file
    except FileNotFoundError:
        pass
    except IndexError:
        break
#os.system('pause')
#os.sys.exit()



### Saving Variables ###

# The file path to save the data
save_file = './model.ckpt'

# Two Tensor Variables: weights and bias
weights = tf.Variable(tf.truncated_normal([2, 3]), name='weights_0')
bias = tf.Variable(tf.truncated_normal([3]), name='bias_0')

# Print the name of Weights and Bias
print('Save Weights: {}'.format(weights.name))
print('Save Bias: {}'.format(bias.name))

# Class used to save and/or restore Tensor Variables
saver = tf.train.Saver()

with tf.Session() as s:
    # Initialize all the Variables
    s.run(tf.global_variables_initializer())

    # Show the values of weights and bias
    print('Weights: {}'.format(s.run(weights)))
    print('Bias: {}'.format(s.run(bias)))

    # Save the model
    saver.save(s, save_file)

print('')


### Loading Variables ###

# The file path to load the data
load_file = './model.ckpt'

# Remove the previous weights and bias
tf.reset_default_graph()

# Two Tensor Variables: weights and bias
weights = tf.Variable(tf.truncated_normal([2, 3]), name='weights_0')
bias = tf.Variable(tf.truncated_normal([3]), name='bias_0')

# Print the name of Weights and Bias
print('Load Weights: {}'.format(weights.name))
print('Load Bias: {}'.format(bias.name))

# Class used to save and/or restore Tensor Variables
saver = tf.train.Saver()

with tf.Session() as s:
    # Load the weights and bias
    saver.restore(s, load_file)

    # Show the values of weights and bias
    print('Weights: {}'.format(s.run(weights)))
    print('Bias: {}'.format(s.run(bias)))

print('\n\n')
#os.sys.exit()



### Save a Trained Model ###

## The model

# Remove previous Tensors and Operations
tf.reset_default_graph()

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

learning_rate = 0.001
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Import MNIST data
mnist = input_data.read_data_sets('.', one_hot=True)
print()

# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

# Weights & bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

# Logits: xW + b
logits = tf.add(tf.matmul(features, weights), bias)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## Let's train that model, then save the weights

import math

save_file = './train_model.ckpt'
batch_size = 128
n_epochs = 100

saver = tf.train.Saver()

# Step for printing and saving.
step = 10

# Launch the graph
with tf.Session() as s:
    s.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(n_epochs + 1):   # '+ 1' is to include the 100-th epoch, too, for printing and saving.
        num_batches = math.ceil(mnist.train.num_examples / batch_size)

        # Loop over all batches
        for i in range(num_batches):
            batch_features, batch_labels = mnist.train.next_batch(batch_size)
            s.run(
                optimizer,
                feed_dict={features: batch_features, labels: batch_labels})

        # Print status for every step epochs
        if epoch % step == 0:
            valid_accuracy = s.run(
                accuracy,
                feed_dict={
                    features: mnist.validation.images,
                    labels: mnist.validation.labels})
            print('Epoch {:<3} - Validation Accuracy: {}'.format(epoch, valid_accuracy))

            # Save the model every step epochs
            saver.save(s, save_file, global_step=epoch)
            print('Trained Model Saved; epoch = {}'.format(epoch))

print('Finished training.\n')


### Load a Trained Model ###

## Let's load the weights and bias from file, then check the test accuracy.

load_file = './train_model.ckpt-{}'.format(n_epochs)        # 100; Test accuracy: 0.734
#load_file = './train_model.ckpt-{}'.format(n_epochs - 40)   # 60; Test accuracy: 0.644
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as s:
    saver.restore(s, load_file)

    test_accuracy = s.run(
        accuracy,
        feed_dict={
            features: mnist.test.images,
            labels: mnist.test.labels})

print('Test accuracy: {}'.format(test_accuracy))

