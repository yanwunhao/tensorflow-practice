import tensorflow as tf

# Initialization of Tensors
x = tf.constant(4, shape=(1, 1), dtype=tf.float32)
y = tf.constant([[1, 2, 3], [4, 5, 6]])
print(x, y)

ones = tf.ones((3, 3))
zeros = tf.zeros((2, 3))
identity = tf.eye(3)  # tf.eye() for identity matrix
print(ones, zeros, identity)

norm = tf.random.normal((3, 3), mean=0, stddev=1)  # normal distribution randoms
uniform = tf.random.uniform((1, 3), minval=0, maxval=1)  # uniform distribution randoms
print(norm, uniform)

tf_range_1 = tf.range(9)
tf_range_2 = tf.range(start=1, limit=9, delta=2)  # delta means step, 9 is not included
print(tf_range_1, tf_range_2)

a = tf.constant([[1, 2, 3], [4, 5, 6]])
a_float64 = tf.cast(a, dtype=tf.float64)  # convert datatype of variable
print(a_float64)
