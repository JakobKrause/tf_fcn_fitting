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

        with tf.GradientTape() as tapeX:
            with tf.GradientTape() as tape:
                tapeX.watch(x)
                y_pred = self(x, training=True)  # Forward pass
                # Compute our own loss
                dy_dx = tapeX.gradient(y_pred, x)
                loss = keras.losses.mean_squared_error(y, dy_dx)

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
x=x.reshape(-1,1)
# Define the original function y = x^2 / 3
def original_function(x):
    return x**2 / 3

# Compute the integral of the function
def integral_function(x):
    return (x**3)/9
y_original = original_function(x)
y_integral = integral_function(x)

model.fit(x, y_original, epochs=2000, callbacks=[lr_scheduler])

# Define the network prediction
def network_prediction(x, model):
    return model.predict(x)
def derivative_network_prediction(x, model):
    with tf.GradientTape() as tapeXX:
        tapeXX.watch(x)
        y_pred = model(x)
    return tapeXX.gradient(y_pred, x)

y_pred = network_prediction(x, model)
derivative_y_pred = derivative_network_prediction(tf.constant(x), model)


plt.figure(figsize=(10, 6))
plt.plot(x, y_original, label='Original Function: y = x^2 / 3')
plt.plot(x, y_integral, label='Integral of Original Function')
plt.plot(x, y_pred, label='Network Prediction')
plt.plot(x, derivative_y_pred, label='Derivative of Network Prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.title('Original Function, Integral, Network Prediction, and Derivative of Prediction')
plt.show()
