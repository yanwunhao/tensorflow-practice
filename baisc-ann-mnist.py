import tensorflow as tf
from tensorflow.keras.datasets import mnist

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# Flatten
# Shape dimension can be -1.
# In this case, the value is inferred from the length of the array and remaining dimensions.
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.  # -1 means keeping previous dimension
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Sequential API (Very convenient, not very flexible)
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10)
    ]
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=["accuracy"]
)

model.fit(x_train, y_train, batch_size=128, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=128, verbose=2)
