import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt

class CustomModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute our own loss
            loss = keras.losses.mean_squared_error(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(y, y_pred)
        return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.mae_metric]


# Construct and compile an instance of CustomModel
input = tf.keras.layers.Input(shape=(1,))
x = tf.keras.layers.Dense(15, activation='tanh',  use_bias=True,  input_shape=(2,))(input)
x = tf.keras.layers.Dense(8, activation='tanh',  use_bias=True,  input_shape=(15,))(x)
#x = tf.keras.layers.Dense(10, activation='tanh',  use_bias=True, input_shape=(10,))(x)
output = tf.keras.layers.Dense(1,  use_bias=True, input_shape=(8,))(x)
model = CustomModel(input, output)

model.compile(optimizer="adam")
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=100, min_lr=0.0001)

x = np.arange(-1, 1, 0.01)
y = pow(x,2)/3

model.fit(x, y, epochs=300, callbacks=[lr_scheduler])

fig = plt.figure()
plt.plot(x,y)
plt.plot(x, model.predict(x))
plt.show()
