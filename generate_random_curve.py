import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import csv

def generate_random_3d_curve_with_curvature(N, l, max_curve):
    x = np.zeros(N + 1)
    y = np.zeros(N + 1)
    z = np.zeros(N + 1)

    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, np.pi)
    dx = l * np.sin(phi) * np.cos(theta)
    dy = l * np.sin(phi) * np.sin(theta)
    dz = l * np.cos(phi)
    directions = np.array([[dx, dy, dz]])

    for i in range(1, N + 1):
        max_angle = l * max_curve
        angle = np.random.uniform(-max_angle, max_angle)
        if np.all(directions[-1] == 0):
            axis = np.array([1, 0, 0])
        else:
            axis = np.random.randn(3)
            axis -= axis.dot(directions[-1]) * directions[-1] / np.linalg.norm(directions[-1])**2
            axis /= np.linalg.norm(axis)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K.dot(K)
        new_direction = R.dot(directions[-1])
        directions = np.vstack([directions, new_direction])
        x[i] = x[i - 1] + new_direction[0]
        y[i] = y[i - 1] + new_direction[1]
        z[i] = z[i - 1] + new_direction[2]
    
    return x, y, z

def compute_distance(x, y, z):
    return np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2 + (z[-1] - z[0])**2)

def compute_x_distance(x):
    return abs(x[-1] - x[0])

def compute_y_distance(y):
    return abs(y[-1] - y[0])

def compute_z_distance(z):
    return abs(z[-1] - z[0])

def compute_area_projection_true(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def compute_area_projection(x,y):
    A = abs(np.trapz(y, x))  # calculate the area using trapezoidal integration
    return A

def calculate_mean_values(N, l, max_curve, M):
    distances = []
    areas_xy = []
    areas_yz = []
    areas_xz = []

    for _ in range(M):
        x, y, z = generate_random_3d_curve_with_curvature_PS(N, l, max_curve)
        distances.append(compute_distance(x, y, z))
        areas_xy.append(compute_area_projection(x, y))
        areas_yz.append(compute_area_projection(y, z))
        areas_xz.append(compute_area_projection(x, z))

    mean_distance = np.mean(distances)
    mean_area_xy = np.mean(areas_xy)
    mean_area_yz = np.mean(areas_yz)
    mean_area_xz = np.mean(areas_xz)

    # Plot histograms
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 2, 1)
    plt.hist(distances, bins=20, edgecolor='k')
    plt.title('Histogram of Distances')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 2)
    plt.hist(areas_xy, bins=20, edgecolor='k')
    plt.title('Histogram of XY Projection Areas')
    plt.xlabel('Area XY')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 3)
    plt.hist(areas_yz, bins=20, edgecolor='k')
    plt.title('Histogram of YZ Projection Areas')
    plt.xlabel('Area YZ')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 4)
    plt.hist(areas_xz, bins=20, edgecolor='k')
    plt.title('Histogram of XZ Projection Areas')
    plt.xlabel('Area XZ')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    return mean_distance, mean_area_xy, mean_area_yz, mean_area_xz

def calculate_values_vs_N(N_start, N_end, l, max_curve, I, num_steps):
    mean_distances = []
    mean_x_distances = []
    mean_y_distances = []
    mean_z_distances = []
    mean_areas_xy = []
    mean_areas_yz = []
    mean_areas_xz = []

    N_values = np.round(np.linspace(N_start, N_end, num_steps)).astype(int)
    
    for N in N_values:
        distances = []
        x_distances = []
        y_distances = []
        z_distances = []
        areas_xy = []
        areas_yz = []
        areas_xz = []

        for _ in range(I):
            x, y, z = generate_random_3d_curve_with_curvature_PS(N, l, max_curve)
            distances.append(compute_distance(x, y, z))
            x_distances.append(compute_x_distance(x))
            y_distances.append(compute_y_distance(y))
            z_distances.append(compute_z_distance(z))
            areas_xy.append(compute_area_projection(x, y))
            areas_yz.append(compute_area_projection(y, z))
            areas_xz.append(compute_area_projection(x, z))

        mean_distance = np.mean(distances)
        mean_x_distance = np.mean(x_distances)
        mean_y_distance = np.mean(y_distances)
        mean_z_distance = np.mean(z_distances)
        mean_area_xy = np.mean(areas_xy)
        mean_area_yz = np.mean(areas_yz)
        mean_area_xz = np.mean(areas_xz)

        mean_distances.append(mean_distance)
        mean_x_distances.append(mean_x_distance)
        mean_y_distances.append(mean_y_distance)
        mean_z_distances.append(mean_z_distance)
        mean_areas_xy.append(mean_area_xy)
        mean_areas_yz.append(mean_area_yz)
        mean_areas_xz.append(mean_area_xz)
        print(N)

    # Plot the results
    plt.figure(figsize=(14, 8))
    
    #popt_log, pcov = curve_fit(logarithmic_function, N_values, mean_distances)
    #a_fit_log, b_fit_log, c_fit_log, d_fit_log = popt_log
    #mean_distances_log_fit = logarithmic_function(N_values, a_fit_log, b_fit_log, c_fit_log, d_fit_log)
    #popt_root, pcov = curve_fit(root_function, N_values, mean_distances)
    #a_root, b_root, c_root, d_root = popt_root
    #mean_distances_root_fit = root_function(N_values, a_root, b_root, c_root, d_root)
    plt.subplot(2, 2, 1)
    plt.plot(N_values, mean_distances, marker='o', label='Mean Distances')  #log or root
    #plt.plot(N_values, mean_distances_log_fit, linestyle='--', color='red', label='Fitted Logarithmic Curve')
    #plt.plot(N_values, mean_distances_root_fit, linestyle='--', color='green', label='Fitted Square Root Curve')
    plt.title('Mean Distance vs N')
    plt.xlabel('N')
    plt.ylabel('Mean Distance')
    plt.legend()

    #coefficients_areas_xy = np.polyfit(N_values, mean_areas_xy, 1)
    #poly_fit_areas_xy = np.poly1d(coefficients_areas_xy)
    plt.subplot(2, 2, 2)
    plt.plot(N_values, mean_areas_xy, marker='o', label='Mean Area XY')   #linear
    #plt.plot(N_values, poly_fit_areas_xy(N_values), linestyle='--', color='red', label='Linear Fit')
    plt.title('Mean XY Projection Area vs N')
    plt.xlabel('N')
    plt.ylabel('Mean Area XY')
    plt.legend()

    #coefficients_areas_yz = np.polyfit(N_values, mean_areas_yz, 1)
    #poly_fit_areas_yz = np.poly1d(coefficients_areas_yz)
    plt.subplot(2, 2, 3)
    plt.plot(N_values, mean_areas_yz, marker='o', label='Mean Area YZ')   #linear
    #plt.plot(N_values, poly_fit_areas_yz(N_values), linestyle='--', color='red', label='Linear Fit')
    plt.title('Mean YZ Projection Area vs N')
    plt.xlabel('N')
    plt.ylabel('Mean Area YZ')
    plt.legend()

    coefficients_areas_xz = np.polyfit(N_values, mean_areas_xz, 1)
    poly_fit_areas_xz = np.poly1d(coefficients_areas_xz)
    plt.subplot(2, 2, 4)
    plt.plot(N_values, mean_areas_xz, marker='o', label='Mean Area XZ')   #linear
    #plt.plot(N_values, poly_fit_areas_xz(N_values), linestyle='--', color='red', label='Linear Fit')
    plt.title('Mean XZ Projection Area vs N')
    plt.xlabel('N')
    plt.ylabel('Mean Area XZ')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return mean_distances, mean_x_distances, mean_y_distances, mean_z_distances, mean_areas_xy, mean_areas_yz, mean_areas_xz

def logarithmic_function(x, a, b, c, d):
    return a * np.log(b * x + c) + d

def root_function(x, a, b, c, d):
    return a * np.sqrt(b * x + c) + d





#pulse sequence method of curve generating
def  generate_random_3d_curve_with_curvature_PS(N,l=1,maximumCurve=10000):
    T=0.0000005
    offset=0
    # Generate the first column with values between 0 and 100
    first_column = np.random.uniform(0, 0, N)
    # Generate the second column with values between -180 and 180
    second_column = np.random.uniform(-180, 180, N)
    # Combine both columns to create the N x 2 array
    PS = np.column_stack((first_column, second_column))
    maximumAmplitude = maximumCurve/(2*np.pi)
    initialVector=np.array([0,0,1])
    CM=createCoordinates_Matrix(PS,T,l,maximumAmplitude, offset, initialVector) #A Matrix which contains the Coordinate of Points of the Curve is created
    return CM[:,0], CM[:,1], CM[:,2]

def createCoordinates_Matrix(PS,T,l,Umax,offset,initialVector):

    VM = createVectors_Matrix(PS.astype(np.float64),np.float64(T),np.float64(l),np.float64(Umax),np.float64(offset),initialVector.astype(np.float64))

    h = np.ones((np.shape(VM)[0]+1,3))
    cn=np.array([0,0,0])      #first Coordinates
    h[0,:]=cn
    for n in range(np.shape(VM)[0]):
        cn=cn+VM[n,:]      #Coordinates = Coordinates of point befor + the fitting Vector from VM
        h[n+1,:]=cn
    CM=h
    return CM

def createVectors_Matrix(PS, T, l, Umax, offset, initialVector):
    num_vectors = len(PS) + 1
    m = np.empty((num_vectors, 3))

    m[0] = l * initialVector

    angle_factor = -2 * np.pi * T
    Umax_factor = Umax / 100

    vn = m[0]

    for i, v in enumerate(PS):
        Ux = v[0] * Umax_factor * np.cos(np.radians(v[1]))
        Uy = v[0] * Umax_factor * np.sin(np.radians(v[1]))
        Uz = offset

        n = np.array([Ux, Uy, Uz])

        norm_n = np.linalg.norm(n)
        if norm_n != 0:
            n /= norm_n
            cosa = np.cos(angle_factor * norm_n)
            sina = np.sin(angle_factor * norm_n)
            mcosa = 1 - cosa
            Rn = np.array([[n[0]**2 * mcosa + cosa, n[0] * n[1] * mcosa - n[2] * sina, n[0] * n[2] * mcosa + n[1] * sina],
                           [n[0] * n[1] * mcosa + n[2] * sina, n[1]**2 * mcosa + cosa, n[1] * n[2] * mcosa - n[0] * sina],
                           [n[2] * n[0] * mcosa - n[1] * sina, n[2] * n[1] * mcosa + n[0] * sina, n[2]**2 * mcosa + cosa]])
        else:
            Rn = np.eye(3)

        vn = np.dot(Rn, vn)
        m[i + 1] = vn
    return m





# Parameters
N = 1000  # Number of line segments
l = 0.0000005  # Length of each segment
max_curve = 2*np.pi*10000  # Maximal curvature in radians
M = 100  # Number of simulations for mean value calculation

#plot curve and show data
x, y, z = generate_random_3d_curve_with_curvature_PS(N, l, max_curve)

distance = compute_distance(x, y, z)

area_xy = compute_area_projection(x, y)
area_yz = compute_area_projection(y, z)
area_xz = compute_area_projection(x, z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Random 3D Curve with Curvature Constraint')

print(f'Cartesian distance between endpoints: {distance:.2f}')
print(f'Area enclosed by XY projection: {area_xy:.2f}')
print(f'Area enclosed by YZ projection: {area_yz:.2f}')
print(f'Area enclosed by XZ projection: {area_xz:.2f}')

plt.show()

# Calculate mean values
mean_distance, mean_area_xy, mean_area_yz, mean_area_xz = calculate_mean_values(N, l, max_curve, M)
print(f'Mean distance: {mean_distance:.2f}')
print(f'Mean XY projection area: {mean_area_xy:.2f}')
print(f'Mean YZ projection area: {mean_area_yz:.2f}')
print(f'Mean XZ projection area: {mean_area_xz:.2f}')

# Calculate values vs N
N_start = 10
N_end = 1305
num_steps = 40
I = 25  # Number of simulations per value for N in vs N calculation
distances, x_distances, y_distances, z_distances, areas_xy, areas_yz, areas_xz = calculate_values_vs_N(N_start, N_end, l, max_curve, I, num_steps)
duration =  np.round(np.linspace(N_start, N_end, num_steps)).astype(int)*l


# Define a function to write a single list to a CSV file
def write_list_to_csv(filename, header, data):
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        csvwriter.writerow([header])
        # Write the data
        for item in data:
            csvwriter.writerow([item])

# Write each list to a separate CSV file
#write_list_to_csv('/Users/leon/Desktop/Physik/Glaser/Bachelor_Thesis/random_generated_curves/PSdistances.csv', 'distances_in_sec', distances)
#write_list_to_csv('/Users/leon/Desktop/Physik/Glaser/Bachelor_Thesis/random_generated_curves/x_PSdistances.csv', 'x_distances_in_sec', x_distances)
#write_list_to_csv('/Users/leon/Desktop/Physik/Glaser/Bachelor_Thesis/random_generated_curves/y_PSdistances.csv', 'y_distances_in_sec', y_distances)
#write_list_to_csv('/Users/leon/Desktop/Physik/Glaser/Bachelor_Thesis/random_generated_curves/z_PSdistances.csv', 'z_distances_in_sec', z_distances)
#write_list_to_csv('/Users/leon/Desktop/Physik/Glaser/Bachelor_Thesis/random_generated_curves/PS_approach/PSareas_xy.csv', 'xy_areas_in_sec_squared', areas_xy)
#write_list_to_csv('/Users/leon/Desktop/Physik/Glaser/Bachelor_Thesis/random_generated_curves/PS_approach/PSareas_yz.csv', 'yz_areas_in_sec_squared', areas_yz)
#write_list_to_csv('/Users/leon/Desktop/Physik/Glaser/Bachelor_Thesis/random_generated_curves/PS_approach/PSareas_xz.csv', 'xz_areas_in_sec_squared', areas_xz)
#write_list_to_csv('/Users/leon/Desktop/Physik/Glaser/Bachelor_Thesis/random_generated_curves/PSduration', 'duration_in_sec', duration)

