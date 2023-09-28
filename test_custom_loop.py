import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

###########################################################################################################################
Training = True  # Set to True for training, False for loading pre-trained model
###########################################################################################################################

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
    

###########################################################################################################################
#############################################Data import###################################################################
###########################################################################################################################
# Function to read the data from the text file
powerscale = 1
numInputs = 2
numOutputs = 1

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
            z.append(float(values[2]))

    return np.array(x_y), np.array(z)

# File path to your data file
file_path = 'ANN_GaAs_Id_xx.dat'  # Replace with the actual file path

# Read the dataPP
x_y, z = read_data(file_path)



# Truncate
x_y = x_y[76:418]
z=z[76:418]##*100

# Apply scaling, for better convergence
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(x_y)
x_y = scaler.transform(x_y)

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
    output = tf.keras.layers.Dense(1,  use_bias=True, input_shape=(8,))(x)
    model = CustomModel(input, output)
    #model = tf.keras.models.Model(input, output)

    # Compile the model
    initial_learning_rate = 0.1
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=adam_optimizer)

    # Train the model
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=50, min_lr=0.0001)
    model.fit(x_y, z, epochs=300, callbacks=[lr_scheduler])
    model.save("my_model.keras")
else:
    # Load the pre-trained model
    model = tf.keras.models.load_model('my_model.keras')

print(model.summary())




###########################################################################################################################
#############################################Function generation###########################################################
###########################################################################################################################

# Extract weights and biases from the trained model
w_b_list = [layer.get_weights() for layer in model.layers]
weights = []
biases = []
for element in w_b_list:
    if element:
        weights.append(element[0]) 
        biases.append(element[1]) 
    #print(element)

# weights = [element[0] for element in w_b_array]
# biases = [element[1] for element in w_b_array]

# Activation functions for each layer
#activation_functions = [layer.get_config()['activation'] for layer in model.layers]
activation_functions = []
config_list = [layer for layer in model.layers]
for element in config_list:
    config = element.get_config()
    if (element._object_identifier != '_tf_keras_input_layer'):
        activation_functions.append(element.activation.__name__)

# Print the extracted information
for i in range(len(weights)):
    print(f"Layer {i+1}:")
    print(f"Weights:\n{weights[i]}")
    print(f"Biases:\n{biases[i]}")
    print(f"Activation Function: {activation_functions[i]}\n")

# Parse function to txt output
# outputFunction = []
# for i in range(1, numInputs+1):


###########################################################################################################################
#############################################Plotting######################################################################
###########################################################################################################################
# Plot the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#dump = np.reshape(model.predict(x_y),(342))
# ax.scatter(x, y, z, c=model.predict(x_y), cmap='viridis')
ax.scatter(x_y[:, 0], x_y[:, 1], pow(model.predict(x_y), powerscale))
ax.scatter(x_y[:, 0], x_y[:, 1], pow(z, powerscale))
ax.scatter(x_y[:, 0], x_y[:, 1], (pow(np.reshape(model.predict(x_y),(342)), powerscale)-pow(z, powerscale)))
plt.show()