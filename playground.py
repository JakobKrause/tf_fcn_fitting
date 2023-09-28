import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
###########################################################################################################################
#############################################Get network data##############################################################
###########################################################################################################################
# Load model
model = tf.keras.models.load_model('my_model.keras')
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
                w = weights[i][j][k]  # Example: Replace with actual input weights
                argument += f'(v_{j + 1})*{w}'

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
        print(f'Argument {k + 1}: {argument_cell[i][k]}')
v_1_a = np.arange(-1,1,0.1)
v_2 = np.arange(-1,1,0.1)


###########################################################################################################################
#############################################Plotting######################################################################
###########################################################################################################################
# Plot the data
fig = plt.figure()
for v_1 in v_1_a:
    y=(0.009007014334201813+np.tanh(-0.13650643825531006+np.tanh((v_1)*-0.12384790927171707+(v_2)*-0.09657158702611923+1.0468719005584717)*0.5514774322509766+np.tanh((v_1)*0.47452372312545776+(v_2)*0.7280949354171753+-0.5270569324493408)*-0.12256590276956558+np.tanh((v_1)*-0.19500184059143066+(v_2)*0.0021033002994954586+0.5738471746444702)*0.6371038556098938+np.tanh((v_1)*-0.44877785444259644+(v_2)*0.0054846713319420815+0.2965036928653717)*-0.16087467968463898+np.tanh((v_1)*0.10514742136001587+(v_2)*3.9007010459899902+2.3419241905212402)*-0.16237367689609528+np.tanh((v_1)*1.1125930547714233+(v_2)*0.2205936163663864+-1.3314977884292603)*-0.368569016456604+np.tanh((v_1)*-0.9496410489082336+(v_2)*-0.44385021924972534+1.5235673189163208)*0.5390163064002991+np.tanh((v_1)*1.1366666555404663+(v_2)*-1.6501853466033936+0.49311649799346924)*-0.8562876582145691+np.tanh((v_1)*0.21379248797893524+(v_2)*0.2416362762451172+-0.8754026293754578)*-0.34128427505493164+np.tanh((v_1)*-0.3121491074562073+(v_2)*-0.011732502840459347+-0.28258198499679565)*-0.13016411662101746+np.tanh((v_1)*-0.29886147379875183+(v_2)*-0.5015645027160645+0.3868985176086426)*0.4647044241428375+np.tanh((v_1)*-1.2829307317733765+(v_2)*1.1696923971176147+1.0578027963638306)*-0.7872620224952698+np.tanh((v_1)*0.39332908391952515+(v_2)*0.11535043269395828+0.5732401609420776)*0.3430286943912506+np.tanh((v_1)*-0.6353321671485901+(v_2)*-0.27179059386253357+0.7551178932189941)*0.6706275343894958+np.tanh((v_1)*-0.9731895923614502+(v_2)*-0.0007533314637839794+0.64208984375)*0.674302339553833)*-0.29451948404312134+np.tanh(-0.39705145359039307+np.tanh((v_1)*-0.12384790927171707+(v_2)*-0.09657158702611923+1.0468719005584717)*-0.31644418835639954+np.tanh((v_1)*0.47452372312545776+(v_2)*0.7280949354171753+-0.5270569324493408)*-0.14318260550498962+np.tanh((v_1)*-0.19500184059143066+(v_2)*0.0021033002994954586+0.5738471746444702)*-0.2635900676250458+np.tanh((v_1)*-0.44877785444259644+(v_2)*0.0054846713319420815+0.2965036928653717)*0.10292521864175797+np.tanh((v_1)*0.10514742136001587+(v_2)*3.9007010459899902+2.3419241905212402)*0.13523779809474945+np.tanh((v_1)*1.1125930547714233+(v_2)*0.2205936163663864+-1.3314977884292603)*0.21812689304351807+np.tanh((v_1)*-0.9496410489082336+(v_2)*-0.44385021924972534+1.5235673189163208)*-0.27052441239356995+np.tanh((v_1)*1.1366666555404663+(v_2)*-1.6501853466033936+0.49311649799346924)*0.07960919290781021+np.tanh((v_1)*0.21379248797893524+(v_2)*0.2416362762451172+-0.8754026293754578)*-0.5085920095443726+np.tanh((v_1)*-0.3121491074562073+(v_2)*-0.011732502840459347+-0.28258198499679565)*-0.7372334599494934+np.tanh((v_1)*-0.29886147379875183+(v_2)*-0.5015645027160645+0.3868985176086426)*0.25698673725128174+np.tanh((v_1)*-1.2829307317733765+(v_2)*1.1696923971176147+1.0578027963638306)*0.04199172183871269+np.tanh((v_1)*0.39332908391952515+(v_2)*0.11535043269395828+0.5732401609420776)*0.19755719602108002+np.tanh((v_1)*-0.6353321671485901+(v_2)*-0.27179059386253357+0.7551178932189941)*0.5477747917175293+np.tanh((v_1)*-0.9731895923614502+(v_2)*-0.0007533314637839794+0.64208984375)*0.05864347144961357)*-0.07477454096078873+np.tanh(0.16485260426998138+np.tanh((v_1)*-0.12384790927171707+(v_2)*-0.09657158702611923+1.0468719005584717)*0.44189679622650146+np.tanh((v_1)*0.47452372312545776+(v_2)*0.7280949354171753+-0.5270569324493408)*-0.07746794819831848+np.tanh((v_1)*-0.19500184059143066+(v_2)*0.0021033002994954586+0.5738471746444702)*0.6403592824935913+np.tanh((v_1)*-0.44877785444259644+(v_2)*0.0054846713319420815+0.2965036928653717)*-0.3573261499404907+np.tanh((v_1)*0.10514742136001587+(v_2)*3.9007010459899902+2.3419241905212402)*0.665711522102356+np.tanh((v_1)*1.1125930547714233+(v_2)*0.2205936163663864+-1.3314977884292603)*-0.3629808723926544+np.tanh((v_1)*-0.9496410489082336+(v_2)*-0.44385021924972534+1.5235673189163208)*0.7269253730773926+np.tanh((v_1)*1.1366666555404663+(v_2)*-1.6501853466033936+0.49311649799346924)*0.45069456100463867+np.tanh((v_1)*0.21379248797893524+(v_2)*0.2416362762451172+-0.8754026293754578)*-0.20544838905334473+np.tanh((v_1)*-0.3121491074562073+(v_2)*-0.011732502840459347+-0.28258198499679565)*-0.5984093546867371+np.tanh((v_1)*-0.29886147379875183+(v_2)*-0.5015645027160645+0.3868985176086426)*-0.008009903132915497+np.tanh((v_1)*-1.2829307317733765+(v_2)*1.1696923971176147+1.0578027963638306)*0.03676152601838112+np.tanh((v_1)*0.39332908391952515+(v_2)*0.11535043269395828+0.5732401609420776)*0.9982168674468994+np.tanh((v_1)*-0.6353321671485901+(v_2)*-0.27179059386253357+0.7551178932189941)*0.1900498867034912+np.tanh((v_1)*-0.9731895923614502+(v_2)*-0.0007533314637839794+0.64208984375)*-0.0066710300743579865)*0.3624829053878784+np.tanh(0.35366135835647583+np.tanh((v_1)*-0.12384790927171707+(v_2)*-0.09657158702611923+1.0468719005584717)*-0.6223641037940979+np.tanh((v_1)*0.47452372312545776+(v_2)*0.7280949354171753+-0.5270569324493408)*0.5295553803443909+np.tanh((v_1)*-0.19500184059143066+(v_2)*0.0021033002994954586+0.5738471746444702)*-0.31242382526397705+np.tanh((v_1)*-0.44877785444259644+(v_2)*0.0054846713319420815+0.2965036928653717)*0.41003653407096863+np.tanh((v_1)*0.10514742136001587+(v_2)*3.9007010459899902+2.3419241905212402)*0.4403958022594452+np.tanh((v_1)*1.1125930547714233+(v_2)*0.2205936163663864+-1.3314977884292603)*0.8077688217163086+np.tanh((v_1)*-0.9496410489082336+(v_2)*-0.44385021924972534+1.5235673189163208)*0.0636477842926979+np.tanh((v_1)*1.1366666555404663+(v_2)*-1.6501853466033936+0.49311649799346924)*-0.474079430103302+np.tanh((v_1)*0.21379248797893524+(v_2)*0.2416362762451172+-0.8754026293754578)*0.3285956382751465+np.tanh((v_1)*-0.3121491074562073+(v_2)*-0.011732502840459347+-0.28258198499679565)*0.3447956144809723+np.tanh((v_1)*-0.29886147379875183+(v_2)*-0.5015645027160645+0.3868985176086426)*-0.5218341946601868+np.tanh((v_1)*-1.2829307317733765+(v_2)*1.1696923971176147+1.0578027963638306)*-0.5585415363311768+np.tanh((v_1)*0.39332908391952515+(v_2)*0.11535043269395828+0.5732401609420776)*-0.5455079078674316+np.tanh((v_1)*-0.6353321671485901+(v_2)*-0.27179059386253357+0.7551178932189941)*0.08772782981395721+np.tanh((v_1)*-0.9731895923614502+(v_2)*-0.0007533314637839794+0.64208984375)*-0.8778740167617798)*0.07451275736093521+np.tanh(0.04079722240567207+np.tanh((v_1)*-0.12384790927171707+(v_2)*-0.09657158702611923+1.0468719005584717)*-1.1073769330978394+np.tanh((v_1)*0.47452372312545776+(v_2)*0.7280949354171753+-0.5270569324493408)*0.5410298109054565+np.tanh((v_1)*-0.19500184059143066+(v_2)*0.0021033002994954586+0.5738471746444702)*-0.8584175705909729+np.tanh((v_1)*-0.44877785444259644+(v_2)*0.0054846713319420815+0.2965036928653717)*0.14258316159248352+np.tanh((v_1)*0.10514742136001587+(v_2)*3.9007010459899902+2.3419241905212402)*-0.06210242211818695+np.tanh((v_1)*1.1125930547714233+(v_2)*0.2205936163663864+-1.3314977884292603)*-0.11368193477392197+np.tanh((v_1)*-0.9496410489082336+(v_2)*-0.44385021924972534+1.5235673189163208)*0.04799346625804901+np.tanh((v_1)*1.1366666555404663+(v_2)*-1.6501853466033936+0.49311649799346924)*-1.0817933082580566+np.tanh((v_1)*0.21379248797893524+(v_2)*0.2416362762451172+-0.8754026293754578)*1.0425446033477783+np.tanh((v_1)*-0.3121491074562073+(v_2)*-0.011732502840459347+-0.28258198499679565)*0.3024570047855377+np.tanh((v_1)*-0.29886147379875183+(v_2)*-0.5015645027160645+0.3868985176086426)*-0.39152729511260986+np.tanh((v_1)*-1.2829307317733765+(v_2)*1.1696923971176147+1.0578027963638306)*-0.05401130020618439+np.tanh((v_1)*0.39332908391952515+(v_2)*0.11535043269395828+0.5732401609420776)*-0.5100961327552795+np.tanh((v_1)*-0.6353321671485901+(v_2)*-0.27179059386253357+0.7551178932189941)*-0.008101940155029297+np.tanh((v_1)*-0.9731895923614502+(v_2)*-0.0007533314637839794+0.64208984375)*-0.06453549116849899)*-0.34411191940307617+np.tanh(0.033842314034700394+np.tanh((v_1)*-0.12384790927171707+(v_2)*-0.09657158702611923+1.0468719005584717)*-0.7282114028930664+np.tanh((v_1)*0.47452372312545776+(v_2)*0.7280949354171753+-0.5270569324493408)*0.8665738701820374+np.tanh((v_1)*-0.19500184059143066+(v_2)*0.0021033002994954586+0.5738471746444702)*-0.5247297883033752+np.tanh((v_1)*-0.44877785444259644+(v_2)*0.0054846713319420815+0.2965036928653717)*-0.4848998486995697+np.tanh((v_1)*0.10514742136001587+(v_2)*3.9007010459899902+2.3419241905212402)*2.1812222003936768+np.tanh((v_1)*1.1125930547714233+(v_2)*0.2205936163663864+-1.3314977884292603)*0.9784685969352722+np.tanh((v_1)*-0.9496410489082336+(v_2)*-0.44385021924972534+1.5235673189163208)*-1.7502449750900269+np.tanh((v_1)*1.1366666555404663+(v_2)*-1.6501853466033936+0.49311649799346924)*-0.07701475918292999+np.tanh((v_1)*0.21379248797893524+(v_2)*0.2416362762451172+-0.8754026293754578)*0.43251919746398926+np.tanh((v_1)*-0.3121491074562073+(v_2)*-0.011732502840459347+-0.28258198499679565)*0.06333030760288239+np.tanh((v_1)*-0.29886147379875183+(v_2)*-0.5015645027160645+0.3868985176086426)*-0.3454601466655731+np.tanh((v_1)*-1.2829307317733765+(v_2)*1.1696923971176147+1.0578027963638306)*-2.240556001663208+np.tanh((v_1)*0.39332908391952515+(v_2)*0.11535043269395828+0.5732401609420776)*-0.5450987815856934+np.tanh((v_1)*-0.6353321671485901+(v_2)*-0.27179059386253357+0.7551178932189941)*-0.8941838145256042+np.tanh((v_1)*-0.9731895923614502+(v_2)*-0.0007533314637839794+0.64208984375)*-0.9231225252151489)*0.21709689497947693+np.tanh(0.12257906794548035+np.tanh((v_1)*-0.12384790927171707+(v_2)*-0.09657158702611923+1.0468719005584717)*-0.4664221405982971+np.tanh((v_1)*0.47452372312545776+(v_2)*0.7280949354171753+-0.5270569324493408)*0.8322482109069824+np.tanh((v_1)*-0.19500184059143066+(v_2)*0.0021033002994954586+0.5738471746444702)*-0.07329925894737244+np.tanh((v_1)*-0.44877785444259644+(v_2)*0.0054846713319420815+0.2965036928653717)*0.22797325253486633+np.tanh((v_1)*0.10514742136001587+(v_2)*3.9007010459899902+2.3419241905212402)*3.1709537506103516+np.tanh((v_1)*1.1125930547714233+(v_2)*0.2205936163663864+-1.3314977884292603)*0.10943274199962616+np.tanh((v_1)*-0.9496410489082336+(v_2)*-0.44385021924972534+1.5235673189163208)*-1.3122594356536865+np.tanh((v_1)*1.1366666555404663+(v_2)*-1.6501853466033936+0.49311649799346924)*0.10793755948543549+np.tanh((v_1)*0.21379248797893524+(v_2)*0.2416362762451172+-0.8754026293754578)*0.6428055763244629+np.tanh((v_1)*-0.3121491074562073+(v_2)*-0.011732502840459347+-0.28258198499679565)*0.6620839834213257+np.tanh((v_1)*-0.29886147379875183+(v_2)*-0.5015645027160645+0.3868985176086426)*-0.6435528993606567+np.tanh((v_1)*-1.2829307317733765+(v_2)*1.1696923971176147+1.0578027963638306)*-2.008533000946045+np.tanh((v_1)*0.39332908391952515+(v_2)*0.11535043269395828+0.5732401609420776)*-0.46523764729499817+np.tanh((v_1)*-0.6353321671485901+(v_2)*-0.27179059386253357+0.7551178932189941)*-0.5862147212028503+np.tanh((v_1)*-0.9731895923614502+(v_2)*-0.0007533314637839794+0.64208984375)*0.5089229941368103)*0.18583610653877258+np.tanh(0.33438441157341003+np.tanh((v_1)*-0.12384790927171707+(v_2)*-0.09657158702611923+1.0468719005584717)*-0.6328921914100647+np.tanh((v_1)*0.47452372312545776+(v_2)*0.7280949354171753+-0.5270569324493408)*0.18050852417945862+np.tanh((v_1)*-0.19500184059143066+(v_2)*0.0021033002994954586+0.5738471746444702)*-0.788469135761261+np.tanh((v_1)*-0.44877785444259644+(v_2)*0.0054846713319420815+0.2965036928653717)*-1.094639778137207+np.tanh((v_1)*0.10514742136001587+(v_2)*3.9007010459899902+2.3419241905212402)*0.5152209401130676+np.tanh((v_1)*1.1125930547714233+(v_2)*0.2205936163663864+-1.3314977884292603)*0.902161717414856+np.tanh((v_1)*-0.9496410489082336+(v_2)*-0.44385021924972534+1.5235673189163208)*-0.5155398845672607+np.tanh((v_1)*1.1366666555404663+(v_2)*-1.6501853466033936+0.49311649799346924)*0.490148663520813+np.tanh((v_1)*0.21379248797893524+(v_2)*0.2416362762451172+-0.8754026293754578)*0.4491927921772003+np.tanh((v_1)*-0.3121491074562073+(v_2)*-0.011732502840459347+-0.28258198499679565)*0.8745811581611633+np.tanh((v_1)*-0.29886147379875183+(v_2)*-0.5015645027160645+0.3868985176086426)*0.13125967979431152+np.tanh((v_1)*-1.2829307317733765+(v_2)*1.1696923971176147+1.0578027963638306)*-1.0001213550567627+np.tanh((v_1)*0.39332908391952515+(v_2)*0.11535043269395828+0.5732401609420776)*-0.7467532157897949+np.tanh((v_1)*-0.6353321671485901+(v_2)*-0.27179059386253357+0.7551178932189941)*-0.664591372013092+np.tanh((v_1)*-0.9731895923614502+(v_2)*-0.0007533314637839794+0.64208984375)*-1.0735228061676025)*-0.06765346974134445)

    plt.plot(v_2,y)

# # Plot against model.predict to verify generated function
# v_1 = np.arange(-1,1,0.1)
# v_2 = np.arange(-1,1,0.1)
# for v_1_x in v_1:
#     for v_2_x in v_2:
#         input= [[v_1_x, v_2_x]]
#         output = model.predict(input)
#         plt.scatter(v_2_x, output[0][0])

plt.show()
