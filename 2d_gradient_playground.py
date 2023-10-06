import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
CapacitanceFileFormat = True
powerscale = 1
###########################################################################################################################
#############################################Data import###################################################################
###########################################################################################################################
# Function to read the data from the text file
def read_data(file_path):
    # Read the text file and split it into lines
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize empty arrays for: x, y, and z
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

# Create a grid of x1 and x2 values

# Example array with x0 and x1 values
# Extract x0 and x1 arrays
x0 = x_y[::38, 0]  # Extract every other element starting from the first (x0 values)
x1 = x_y[:,1]
x1 = x1[:38]  # Extract every other element starting from the second (x1 values)

mesh_x1, mesh_x0 = np.meshgrid(x1,x0)
mesh_x0 = mesh_x0.flatten()
mesh_x1 = mesh_x1.flatten()

print("x0 array:", x0)
print("x1 array:", x1)

# Define the function a = f(x1, x2)
  # Replace with your actual function f(x1, x2)
z = z[:,0]
z = z.reshape(11,38)
gradient_x0 = np.zeros_like(z)
for i in range(x1.size):
    dump_z = z[:,i]
    gradient_x0[:,i] = np.gradient(dump_z, x0)

gradient_x1 = np.zeros_like(z)
for i in range(x0.size):
    dump_z = z[i,:]
    gradient_x1[i,:] = np.gradient(dump_z, x1)

# gradient_x0 = np.gradient(z, axis=0)

# Plot the 2D function with non-uniform axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
gradient_x0 = gradient_x0.flatten()
ax.scatter(mesh_x0, mesh_x1, gradient_x0)
ax.scatter(mesh_x0, mesh_x1, z, marker='+', color='k')
ax.scatter(mesh_x0, mesh_x1, gradient_x1, marker='v', color='r')
# plt.matshow(z, extent=[x1.min(), x1.max(), x0.max(), x0.min()])


# # Add labels for the axes
# plt.xlabel('X2 Axis Label')
# plt.ylabel('X1 Axis Label')

plt.show()