import tensorflow as tf

# Output depth
k_output = 64

# Image Properties
image_height = 10
image_width = 10
color_channels = 3

# Convolution filter
filter_size_height = 5
filter_size_width = 5

# Input/Image
input = tf.placeholder(
    tf.float32,
    shape=[None, image_height, image_width, color_channels])
print(input)

# Weight and bias
weight = tf.Variable(tf.truncated_normal(
    shape=[filter_size_height, filter_size_width, color_channels, k_output]))
bias = tf.Variable(tf.zeros([k_output]))

# Apply Convolution
conv_layer = tf.nn.conv2d(input, filter=weight, strides=[1, 2, 2, 1], padding='SAME')
print(conv_layer)
# Add bias
conv_layer = tf.nn.bias_add(conv_layer, bias)
print(conv_layer)
# Apply activation function
conv_layer = tf.nn.relu(conv_layer)
print(conv_layer)
# Apply Max Pooling
conv_layer = tf.nn.max_pool(
    conv_layer,
    ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1],
    padding='SAME')
print(conv_layer)   # 'SAME' gives shape (?, 3, 3, 64); but, 'VALID' gives (?, 2, 2, 64).



