import tensorflow as tf

x = tf.constant([1, 2, 3])
y = tf.constant([9, 8, 7])

# add subtract divide
add_xy = tf.add(x, y)
subtract_xy = tf.subtract(x, y)
divide_xy = tf.divide(x, y)
print(add_xy, subtract_xy, divide_xy)

# product of vectors
product = tf.multiply(x, y)
print(product)

dot_product = tf.tensordot(x, y, axes=1)
# equals tf.reduce_sum(tf.multiply(x, y))
print(dot_product)

# exponent calculation
print(dot_product ** 2)

# matrix multiply
a = tf.random.normal((2, 3))
b = tf.random.normal((3, 4))
mul_of_ab = tf.matmul(a, b)
print(mul_of_ab)  # equals a @ b
