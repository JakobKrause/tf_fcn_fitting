import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def write_strings_to_file(string1, filename, powerscale):
    try:
        with open(filename, 'w') as file:
            file.write(f'import numpy as np\nimport matplotlib.pyplot as plt\n')
            file.write(f'def plotNetworkOutput():\n')
            file.write(f'   v_1_a = np.arange(-5,-2.725,0.01)\n')
            file.write(f'   v_2 = np.arange(0,30,0.3)\n')
            file.write(f'   fig= plt.figure()\n')
            file.write(f'   ax = fig.add_subplot(111, projection=\'3d\')\n')
            file.write(f'   for v_1 in v_1_a:\n')
            if powerscale ==1:
                file.write(f'       y={string1}\n')
            else:
                file.write(f'       y=pow({string1},{powerscale})\n')
            file.write(f'       ax.scatter(v_1, v_2, y , color=\'g\')\n')
            file.write(f'plt.show()')
        print(f'Successfully wrote strings to {filename}')
    except Exception as e:
        print(f'An error occurred: {str(e)}')



def generateVerilogA(model, minIn, maxIn, minOut, maxOut, powerscale):
    ###########################################################################################################################
    #############################################Get network data##############################################################
    ###########################################################################################################################
    # Load model
    #model = tf.keras.models.load_model('my_model.keras')
    # Get activation functions
    activation_functions = []
    config_list = [layer for layer in model.layers]
    for element in config_list:
        config = element.get_config()
        if (element._object_identifier != '_tf_keras_input_layer'): # Input layer is also present. And nodes are not to be calculated
            activation_functions.append(element.activation.__name__)

    # Get weights and biases 
    w_b_list = [layer.get_weights() for layer in model.layers]
    weights = []
    biases = []
    for element in w_b_list:
        if element: # Input layer has no weights / biases
            weights.append(element[0]) 
            biases.append(element[1])

    # Get num_inputs / num_layers
    if(len(activation_functions) == len(weights) == len(biases)):
        num_layers = len(weights)
    else:
        print('Something went wrong with the lenghts of functions/weights/biases vectors')

    num_inputs = len(weights[0])
    ###########################################################################################################################
    #############################################Data import###################################################################
    ###########################################################################################################################
    # Function to read the data from the text file
    #powerscale = 1

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
    #############################################Loop over layers##############################################################
    ###########################################################################################################################
    argument_cell=[]
    for i in range(num_layers):
        print('\nLayer:', i+1)
        layer_argument_cell = []
        # Get transfer function
        t_fcn = activation_functions[i]

        # Check transfer function type
        if t_fcn == 'linear':
            t_fcn = ''
        elif t_fcn =='tanh':
            t_fcn = 'np.tanh'
        

        print('Transfer function:', t_fcn)

        # Get dimension
        dimension = biases[i].size  # Example: Replace with actual dimension
        print('Dimension:', dimension)

        # Loop over dimension
        for k in range(dimension):
            argument = ''

            # Check input
            if i==0: # Only inputs for first layer
                for j in range(num_inputs):
                    if argument != "":
                        argument += '+'
                    w = weights[i][j][k]  # Example: Replace with actual input weights [2.0 * (x - min_x) / (max_x - min_x) - 1.0]
                    argument += f'((v_{j + 1}-{minIn[j]})*2/({maxIn[j]}-{minIn[j]})-1)*{w}' # With applied input scaling

            # Check bias
            if argument != "":
                argument += '+'
            b = biases[i][k]  # Example: Replace with actual bias values
            argument += f'{b}'

            # Check layer
            if i!=0: # No preceeding layers for first layer
                for il in range(biases[i-1].size): # dimension of preceeding layer 
                    lw = weights[i][il][k]
                    if argument != "":
                        argument += '+'
                    argument += f'{argument_cell[i-1][il]}*{lw}'

            # Transfer function wrapping
            if t_fcn == 'logsig':
                argument = f'1 / (1 + exp({argument}))'
            else:
                argument = f'{t_fcn}({argument})'

            layer_argument_cell.append(argument) # Array for current layer

        argument_cell.append(layer_argument_cell) # Array containing whole network 

        # Print the arguments for the current layer
        for k in range(dimension):
            #print(f'Argument {k + 1}: {argument_cell[i][k]}')
            if i == num_layers-1:
                filename = "output.py"
                write_strings_to_file(argument_cell[i][k], filename, powerscale)
    # v_1_a = np.arange(-1,1,0.1)
    # v_2 = np.arange(-1,1,0.1)

