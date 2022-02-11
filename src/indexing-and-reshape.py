import tensorflow as tf

x = tf.constant([0, 1, 1, 2, 3, 1, 2, 3])

print(x[:])
print(x[1:])  # from one to end
print(x[1:3])  # from one to two

print(x[::2])  # skip every 2 elements
print(x[::-1])  # reverse

indices = tf.constant([0, 3])
x_ind = tf.gather(x, indices)
print(x_ind)  # extract element of specified indexing

y = tf.constant([[1, 2], [3, 4], [5, 6]])
print(y[0, :])
print(y[0:2, :])

z = tf.range(9)

z_reshape = tf.reshape(z, (3, 3))
print(z_reshape)

z_transposed = tf.transpose(z_reshape, perm=[1, 0])
print(z_transposed)
