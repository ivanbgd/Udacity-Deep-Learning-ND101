# Solution is available in the other "solution.py" tab
import tensorflow as tf

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

# TODO: Print cross entropy from session

feed_dict = {softmax: softmax_data, one_hot: one_hot_data}
ce = -tf.reduce_sum(one_hot * tf.log(softmax))

with tf.Session() as s:
    print(s.run(ce, feed_dict=feed_dict))
