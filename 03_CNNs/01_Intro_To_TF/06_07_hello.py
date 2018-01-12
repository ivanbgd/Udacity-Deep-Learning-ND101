import tensorflow as tf

# Create TensorFlow object called tensor, "hello_constant"
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)

print(hello_constant)
print(hello_constant.shape)
print(type(hello_constant))
print()


# A is a 0-dimensional int32 tensor
A = tf.constant(1234) 
# B is a 1-dimensional int32 tensor
B = tf.constant([123,456,789]) 
# C is a 2-dimensional int32 tensor
C = tf.constant([ [123,456,789], [222,333,444] ])

print(A)
print(B)
print(C)

print(A.shape)
print(B.shape)
print(C.shape)


x = tf.placeholder(tf.string)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: "Hi world!"})
    print(output)
    print(output.shape)
    print(output.dtype)


x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

feed_dict={x: "Test", y: 123, z: 45.67}

print()
with tf.Session() as sess:
    output = sess.run(x, feed_dict=feed_dict)
    print(output)
    output = sess.run(y, feed_dict=feed_dict)
    print(output)
    output = sess.run(z, feed_dict=feed_dict)
    print(output)
