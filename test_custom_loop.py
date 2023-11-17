import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from playground import generateVerilogA
import os

###########################################################################################################################
Training = True  # Set to True for training, False for loading pre-trained model
TrainOnDerivative = False
CapacitanceFileFormat = False
powerscale = 1
outputVerilogA = True
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
def read_data():

    # Initialize empty arrays for x, y, and z
    x = [4,6,9,14,19,24,29]
    y = [2,2.5,3,3.5,4,5,6,8,10,12]
    result_matrix =  np.zeros((len(y), len(x)))

    # Loop through each folder
    # Construct the file paths for dat1/iv.dat and dat17/iv.dat
    for i in range(len(y)):
        for j in range(len(x)):
            folder_name = f"C:\SST_BP1\SOIMOS_dox{y[i]}nm_L{x[j]}nm"
            file_path1 = os.path.join( folder_name, 'dat1', 'iv.dat')
            file_path17 = os.path.join( folder_name, 'dat17', 'iv.dat')

            try:
                # Read values from dat1/iv.dat
                data1 = np.loadtxt(file_path1, skiprows=1)  # Skip the header line
                ids_vgs0V = data1[1]
                #value1 = data1[0, 1]  # Assuming the value you want is in the first row, second column
                
                # Read values from dat17/iv.dat
                data17 = np.loadtxt(file_path17, skiprows=2)  # Skip the header line
                ids_vgs8V = data17[1]  # Assuming the value you want is in the first row, second column

                # Append the values to the result matrix
                ids_ratio =ids_vgs8V/ids_vgs0V
                result_matrix[i][j] = ids_ratio
            except Exception as e:
                # Handle the case where the file is not found or the format is incorrect
                print(f'Error reading files in folder {folder_name}: {str(e)}')
    x,y = np.meshgrid(x, y)
    x= x.flatten()
    y=y.flatten()
    x_y = np.matrix([x,y])
    x_y = x_y.transpose()
    result_matrix = np.log10(result_matrix.flatten())

    return np.array(x_y), np.array(result_matrix)


# Read the dataPP
x_y, z = read_data()


# # Truncate
# x_y = x_y[76:418]
# z=z[76:418]##*100

# Apply scaling, for better convergence
# TODO Scaling in function generation
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(x_y)
minIn = scaler.data_min_
maxIn = scaler.data_max_
x_y = scaler.transform(x_y)

# if CapacitanceFileFormat:
#     # scalerZ = MinMaxScaler(feature_range=(-1, 1))
#     # scalerZ.fit(z)
#     # minOut = scalerZ.data_min_
#     # maxOut = scalerZ.data_max_
#     # z = scalerZ.transform(z)
#     print("Blabla")
# else:
z_min = np.min(z)
z_max = np.max(z)
z_scaled = ((z - z_min) / (z_max - z_min)) * (1 - 0) + 0#-1 + 2 * ((z - z_min) / (z_max - z_min))
z_root = np.sign(z_scaled) * np.abs(z_scaled) ** (1/powerscale)

# # Apply powerscale. Better fit for data at low power regimes in case a wide range is provided
# plt.figure(figsize=(8, 6))
# plt.plot(z, label='Original Array')
# plt.plot(z_scaled, label='Original Array (scaled 0 to 1)')
# plt.plot(z_root, label='Array after 3rd root')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.legend()
# plt.title('Original Array and 3rd Root of Scaled Array')
# plt.grid(True)
# plt.show()
###########################################################################################################################
#############################################Network Training##############################################################
###########################################################################################################################
if Training:
    input = tf.keras.layers.Input(shape=(2,))
    x = tf.keras.layers.Dense(8, activation="tanh", use_bias=True, input_shape=(2,))(
        input
    )
    x = tf.keras.layers.Dense(8, activation="tanh", use_bias=True, input_shape=(8,))(x)
    if CapacitanceFileFormat:
        output = tf.keras.layers.Dense(1, use_bias=True, input_shape=(8,))(x)
    else:
        output = tf.keras.layers.Dense(1, use_bias=True, input_shape=(8,))(x)
    #model = CustomModel(input, output)
    model = tf.keras.models.Model(input, output)

    # Compile the model
    initial_learning_rate = 0.1
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=adam_optimizer, loss='mse')

    # Train the model
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.9, patience=90, min_lr=0.000001)
    model.fit(x_y, z_scaled, epochs=2000, callbacks=[lr_scheduler])
    model.save("Qg.keras")
else:
    # Load the pre-trained model
    model = tf.keras.models.load_model("Qg.keras")

print(model.summary())

minOut = z_min
maxOut = z_max
###########################################################################################################################
#############################################Generate Verilog A############################################################
###########################################################################################################################
if outputVerilogA:
    generateVerilogA(model, minIn, maxIn, minOut, maxOut, powerscale)
    from output import plotNetworkOutput
    plotNetworkOutput()

###########################################################################################################################
#############################################Plotting######################################################################
###########################################################################################################################
if CapacitanceFileFormat:
    # Plot the data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    toDisplay = 0
    # dump = np.reshape(model.predict(x_y),(342))
    # ax.scatter(x, y, z, c=model.predict(x_y), cmap='viridis')
    trainingData = pow(z, powerscale)
    trainingData0 = trainingData[:, 0]
    trainingData1 = trainingData[:, 1]
    modelData = pow(model.predict(x_y), powerscale)
    modelData = modelData[:, 0]  # [[]]->[] because the model is one output

    ###############
    modelData = modelData.reshape(11, 38)
    x0 = x_y[::38, 0]  # Extract every other element starting from the first (x0 values)
    x1 = x_y[:, 1]
    x1 = x1[:38]  # Extract every other element starting from the second (x1 values)
    gradient_x0 = np.zeros_like(modelData)
    for i in range(x1.size):
        dump = modelData[:, i]
        gradient_x0[:, i] = np.gradient(dump, x0)

    gradient_x1 = np.zeros_like(modelData)
    for i in range(x0.size):
        dump = modelData[i, :]
        gradient_x1[i, :] = np.gradient(dump, x1)

    ###############

    # modelDataDerivative = calculate_gradient(x_y, modelData)
    # modelDataDerivative = modelDataDerivative[:,toDisplay]
    # xxxx = modelDataDerivative[toDisplay]

    ax.scatter(x_y[:, 0], x_y[:, 1], modelData.flatten(), color="g")
    ax.scatter(x_y[:, 0], x_y[:, 1], trainingData0, marker="+", color="k")
    ax.scatter(x_y[:, 0], x_y[:, 1], gradient_x0.flatten(), marker="v", color="r")
    ax.scatter(x_y[:, 0], x_y[:, 1], trainingData1, marker="o", color="k")
    ax.scatter(x_y[:, 0], x_y[:, 1], gradient_x1.flatten(), marker="x", color="b")

    # ax.scatter(x_y[:, 0], x_y[:, 1], modelDataDerivative-trainingData, marker='x', color='k')
    # ax.scatter(x_y[:, 0], x_y[:, 1], modelDataDerivative, marker='+', color='k')
    # ax.scatter(x_y[:, 0], x_y[:, 1], modelDataDerivative[toDisplay])
    plt.show()
else:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #dump = np.reshape(model.predict(x_y),(342))
    # ax.scatter(x, y, z, c=model.predict(x_y), cmap='viridis')
    result  = model.predict(x_y)
    result_scaled = result * (z_max-z_min)+z_min
    ax.scatter(x_y[:, 0], x_y[:, 1], pow(result_scaled, powerscale), marker="x", color="b")
    ax.scatter(x_y[:, 0], x_y[:, 1], pow(z, powerscale), marker="o", color="k")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    x_check = np.linspace(-1, 1, num=60)
    y_check = np.linspace(-1, 1, num=60)
    x_check,y_check = np.meshgrid(x_check, y_check)
    x_check_flat= x_check.flatten()
    y_check_flat=y_check.flatten()
    x_y_check = np.matrix([x_check_flat,y_check_flat])
    x_y_check = x_y_check.transpose()

    result_check  = model.predict(x_y_check)
    result_check = result_check * (z_max-z_min)+z_min
    ax.contour(x_check, y_check, result_check.reshape(60,60),levels=[5], colors='r', width=10)
    ax.plot_wireframe(x_check, y_check, result_check.reshape(60,60))
    ax.scatter(x_y[:, 0], x_y[:, 1], pow(result_scaled, powerscale), marker="x", color="b")


    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.contour(x_check, y_check, result_check.reshape(60,60),levels=[5], colors='r', width=10)



    plt.show()
    