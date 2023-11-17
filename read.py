import numpy as np
import matplotlib.pyplot as plt

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
            z.append(float(values[2]))

    return np.array(x_y), np.array(z)

# File path to your data file
file_path = 'ANN_GaAs_Id_xx.dat'  # Replace with the actual file path

# Read the data
x_y, z = read_data(file_path)

# Print the extracted data
print("x, y:")
print(x_y)
print("z:")
print(z)


# Extract x, y coordinates
x = x_y[:, 0]
y = x_y[:, 1]

# Plot the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=z, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
