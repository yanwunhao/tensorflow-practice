import tensorflow as tf

from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Flatten
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.  # -1 means keeping previous dimension
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.

# Functional API (Very convenient, not very flexible)
inputs = tf.keras.Input(shape=(28*28), name="input_layer")
x = tf.keras.layers.Dense(512, activation="relu", name="first_layer")(inputs)
x = tf.keras.layers.Dense(256, activation="relu", name="second_layer")(x)
outputs = tf.keras.layers.Dense(10, activation="softmax", name="output_layer")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)

model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
