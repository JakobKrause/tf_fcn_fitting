import tensorflow as tf
import keras
import numpy as np
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
            loss = keras.losses.mean_squared_error(y, y_pred)*100

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


# Construct an instance of CustomModel
inputs = keras.Input(shape=(1,))
x = keras.layers.Dense(15, activation='sigmoid')(inputs)
x = keras.layers.Dense(8, activation = 'sigmoid')(x)
outputs = keras.layers.Dense(1)(x)
model = CustomModel(inputs, outputs)

# We don't passs a loss or metrics here.
model.compile(optimizer="adam")

# Just use `fit` as usual -- you can use callbacks, etc.
x = np.arange(0,10,0.01)
x.sort()
y = x**2


model.fit(x, y, epochs=50)
print(model.layers[1].trainable_weights[1])
print(model.weights[0])
print(model.summary())
plt.plot(x,y)
plt.plot(x, model.predict(x))
plt.show()




# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras import backend as K
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler

# powerscale = 2

# ###########################################################################################################################
# # Function to read the data from the text file
# def read_data(file_path):
#     # Read the text file and split it into lines
#     with open(file_path, 'r') as file:
#         lines = file.readlines()

#     # Initialize empty arrays for x, y, and z
#     x_y = []
#     z = []

#     # Parse each line and extract x, y, and z values
#     for line in lines:
#         # Split the line into x, y, and z values (assuming they are space-separated)
#         values = line.strip().split()
#         if len(values) == 3:
#             x_y.append([float(values[0]), float(values[1])])
#             z.append(float(values[2]))

#     return np.array(x_y), np.array(z)

# # File path to your data file
# file_path = 'ANN_GaAs_Id_xx.dat'  # Replace with the actual file path

# # Read the dataPP
# x_y, z = read_data(file_path)
# scaler = MinMaxScaler(feature_range=(-1,1))
# scaler.fit(x_y)
# x_y = scaler.transform(x_y)
# # scaler.fit(z)
# # z = scaler.transform(z)
# z=pow(z, 1/powerscale)
# ###########################################################################################################################


# # Custom loss function (Mean Squared Error with custom scaling factor)
# def custom_loss(y_true, y_pred):
#     scaling_factor = 1.0  # Adjust this scaling factor as needed
#     return K.mean(K.square(y_true - y_pred) * scaling_factor)

# # Define the neural network model with a custom loss function
# def create_model():
#     model = models.Sequential([
#         layers.Dense(20, activation='tanh', input_shape=(2,)),
#         layers.Dense(10, activation='tanh', input_shape=(15,)),
#         layers.Dense(1)  
#         # Output layer
#     ])
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model

# # Create and compile the model
# model = create_model()

# # Train the model
# model.fit(x_y, z, epochs=300, batch_size=32)
# print(model.summary())
# ###########################################################################################################################


# # plt.plot(x_train, model.predict(x_train))
# # plt.show()

# # ax = fig.add_subplot(projection='3d')
# # # Plot a basic wireframe.
# # ax.plot_contour(x_y[:,0], x_y[:,1], model.predict(x_y))
# # Extract x, y coordinates
# x = x_y[:, 0]
# y = x_y[:, 1]

# # Plot the data
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(x, y, z, c=model.predict(x_y), cmap='viridis')
# ax.scatter(x, y, pow(model.predict(x_y),powerscale))
# ax.scatter(x, y, pow(z,powerscale))
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.show()
