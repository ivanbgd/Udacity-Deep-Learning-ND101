# Solution is available in the other "solution.py" tab
import tensorflow as tf


# TODO: Convert the following to TensorFlow:
x = 10
y = 2
z = x/y - 1

Z = tf.placeholder(tf.int32)
feed_dict = {Z: z}
out = None

with tf.Session() as s:
    out = s.run(Z, feed_dict)

# TODO: Print z from a session
print(out)



# TODO: Convert the following to TensorFlow:
x = 10
y = 2
z = x/y - 1

out = None

with tf.Session() as s:
    out = s.run(tf.subtract(tf.div(x, y), 1))

# TODO: Print z from a session
print(out)



# TODO: Convert the following to TensorFlow:
x = 10
y = 2
z = x/y - 1

X, Y, Z = tf.placeholder(tf.int32), tf.placeholder(tf.int32), tf.placeholder(tf.int32)
feed_dict = {X: x, Y: y}
out = None

with tf.Session() as s:
    Z = tf.subtract(tf.div(X, Y), 1)
    out = s.run(Z, feed_dict)

# TODO: Print z from a session
print(out)



# TODO: Convert the following to TensorFlow:
x = 10
y = 2
z = x/y - 1

X, Y, Z = tf.placeholder(tf.int32), tf.placeholder(tf.int32), None
feed_dict = {X: x, Y: y}
out = None

with tf.Session() as s:
    Z = X // Y - 1
    out = s.run(Z, feed_dict)

# TODO: Print z from a session
print(out)

print("")
print(Z)
print()



# TODO: Convert the following to TensorFlow:
x = tf.constant(10)
y = tf.constant(2)
z = tf.subtract(tf.divide(x, y), tf.cast(tf.constant(1), tf.float64))

# TODO: Print z from a session
with tf.Session() as s:
    out = s.run(z)
    print(out)

