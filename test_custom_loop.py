import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

###########################################################################################################################
Training = True  # Set to True for training, False for loading pre-trained model
TrainOnDerivative = False
CapacitanceFileFormat = True
powerscale = 1
###########################################################################################################################

class CustomModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

    def train_step(self, data):
        x, y = data

        if TrainOnDerivative:
            with tf.GradientTape() as tapeX:
                with tf.GradientTape() as tape:
                    tapeX.watch(x)
                    y_pred = self(x, training=True)  # Forward pass
                    # Loss of target versus derivative of prediction
                    dy_dx = tapeX.gradient(y_pred, x)
                    loss = keras.losses.mean_squared_error(y, dy_dx)
        else:
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)  # Forward pass
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
    

###########################################################################################################################
#############################################Data import###################################################################
###########################################################################################################################
# Function to read the data from the text file
def read_data(file_path):
    # Read the text file and split it into lines
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize empty arrays for x, y, and z
    x_y = []
    z = []

    # Parse each line and extract x, y, and z values
    for line in lines:
        # Split the line into x, y, and z values (assuming they are space-separated)
        values = line.strip().split()
        if len(values) == 3:
            x_y.append([float(values[0]), float(values[1])])
            z.append([float(values[2])])
        elif len(values) == 5:
            x_y.append([float(values[0]), float(values[1])])
            z.append([float(values[3]),float(values[4])])
        else:
            print("Error reading data file!")
            return 0
    return np.array(x_y), np.array(z)

# File path to your data file
file_path = 'ANN_GaAs_Qg_xxx.dat'  # Replace with the actual file path

# Read the dataPP
x_y, z = read_data(file_path)



# # Truncate
# x_y = x_y[76:418]
# z=z[76:418]##*100

# Apply scaling, for better convergence
# TODO Scaling in function generation
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(x_y)
x_y = scaler.transform(x_y)

if CapacitanceFileFormat:
    scalerZ = MinMaxScaler(feature_range=(-1,1))
    scalerZ.fit(z)
    z = scalerZ.transform(z)
else:
    z_min = np.min(z)
    z_max = np.max(z)
    z = (z - z_min) / (z_max - z_min)
#Apply powerscale. Better fit for data at low power regimes in case a wide range is provided
z=pow(z, 1/powerscale)

###########################################################################################################################
#############################################Network Training##############################################################
###########################################################################################################################
if Training: 
    input = tf.keras.layers.Input(shape=(2,))
    x = tf.keras.layers.Dense(15, activation='tanh',  use_bias=True,  input_shape=(2,))(input)
    x = tf.keras.layers.Dense(8, activation='tanh',  use_bias=True,  input_shape=(15,))(x)
    #x = tf.keras.layers.Dense(10, activation='tanh',  use_bias=True, input_shape=(10,))(x)
    if CapacitanceFileFormat:
        output = tf.keras.layers.Dense(2,  use_bias=True, input_shape=(8,))(x)
    else:
        output = tf.keras.layers.Dense(1,  use_bias=True, input_shape=(8,))(x)
    model = CustomModel(input, output)
    #model = tf.keras.models.Model(input, output)

    # Compile the model
    initial_learning_rate = 0.1
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=adam_optimizer)

    # Train the model
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=80, min_lr=0.0001)
    model.fit(x_y, z, epochs=1000, callbacks=[lr_scheduler])
    model.save("Qg.keras")
else:
    # Load the pre-trained model
    model = tf.keras.models.load_model('Qg.keras')

print(model.summary())



###########################################################################################################################
#############################################Plotting######################################################################
###########################################################################################################################
# Plot the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#dump = np.reshape(model.predict(x_y),(342))
# ax.scatter(x, y, z, c=model.predict(x_y), cmap='viridis')
trainingData = pow(z, powerscale)
modelData = pow(model.predict(x_y), powerscale)
toDisplay = 1
ax.scatter(x_y[:, 0], x_y[:, 1], modelData[:, toDisplay])
ax.scatter(x_y[:, 0], x_y[:, 1], trainingData[:, toDisplay])
ax.scatter(x_y[:, 0], x_y[:, 1], modelData[:, toDisplay]-trainingData[:, toDisplay])
plt.show()