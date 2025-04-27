import numpy as np
import matplotlib.pyplot as plt
import os

from rotation_matrix_algorithm import *
from differential_geometry import *
from storing_saving_formatting import *

def pulse_sequence_from_curve(curve, max_amplitude, tau, dir_path):
    """
    Calculate the pulse sequence with given max_amplitude and time_per_subpulse (tau) from space-curve and save the data in a new sub directory of dir_path
    Parameters:
    - curve (np.array): An m x 3 array representing the input curve, where m is the number of points on the curve.
    - max_amp (float): The desired maximum amplitude for the pulse.
    - tau (float): The desired time per pulse.

    Output:
    - pulse sequence (np.array): A pulse sequence as n x 2 array (1. column: percentage of max_amp, 2. column: phase)
    - works best for big m (smooth curves)
    """
    dir_path = get_next_subdirectory(dir_path)
    len_scale = 0.9     #test what works best (usually between 0.1 and 4)
    N1 = round(len(curve[:,1])*len_scale)
    required_max_curvature = 2*np.pi*max_amplitude
    # Use arc length as new parameter for curvature and torsion calculation
    curve_repara, arc_array, curvature, torsion = reparametrize_and_calc_curvature_and_torsion(curve, N1)
    print("step 1/3 done")

    # Normalize curvature for pulse sequence
    curve_max_curvature = np.max(curvature)
    N2 = round(arc_array[-1]*curve_max_curvature / (required_max_curvature*tau))
    curve_repara, arc_array, _, _ = reparametrize_and_calc_curvature_and_torsion(curve*curve_max_curvature / required_max_curvature, N2)
    length = len(arc_array)

    interp_curv_fun = interp1d(np.linspace(0,1,len(curvature)), curvature * required_max_curvature / curve_max_curvature, kind='linear', fill_value="extrapolate")
    curvature = interp_curv_fun(np.linspace(0,1,length))
    interp_tors_fun = interp1d(np.linspace(0,1,len(torsion)), torsion * required_max_curvature / curve_max_curvature, kind='linear', fill_value="extrapolate")
    torsion = interp_tors_fun(np.linspace(0,1,length))
    normalized_curvature = curvature / required_max_curvature
    print("step 2/3 done")
    #plot_space_curve(curve_repara)

    # Compute the integrated torsion (cumulative sum of torsion)
    integrated_torsion = np.cumsum(torsion * np.diff(arc_array, prepend=arc_array[0]))  # Integrated torsion based on arc length
    
    #fig = plt.figure(figsize=(10, 8))
    #ax = fig.add_subplot()
    #ax.plot(arc_array, curvature, label='Curvature')
    #ax.plot(arc_array, integrated_torsion, label='Integrated Torsion')
    #ax.legend()
    #plt.title('Torsion/Curvature')
    #plt.show()

    # Create pulse sequence with normalized curvature and integrated torsion
    pulse_sequence = np.column_stack((100*normalized_curvature, integrated_torsion*180/np.pi))

    x = curve_repara[:,0]
    y = curve_repara[:,1]
    z = curve_repara[:,2]
    # Compute projection areas using trapezoidal integration
    area_xy = np.trapz(y, x)  
    area_xz = np.trapz(z, x)  
    area_yz = np.trapz(z, y)  

    # Save curve
    curve_path = os.path.join(dir_path, 'curve.txt')
    np.savetxt(curve_path, curve_repara, delimiter=',', fmt='%.12f', comments='')

    # Save areas and parameters to a text file
    flip_angle = np.arccos(np.dot(curve_repara[0,:], curve_repara[-1,:])/(np.linalg.norm(curve_repara[0,:])*np.linalg.norm(curve_repara[-1,:])))*180/np.pi
    params_path = os.path.join(dir_path, 'parameters_and_areas.txt')
    with open(params_path, 'w') as f:
        f.write(f"Maximum amplitude: {max_amplitude} kHz\n")
        f.write(f"Time per subpulse: {tau} s\n")
        f.write(f"Number of subpulses: {len(normalized_curvature)}\n")
        f.write(f"Pulse flip angle: {flip_angle}°\n")
        f.write(f"distance of curve ends:\n")
        f.write(f"x distance of curve ends: {abs(curve_repara[-1,0] - curve_repara[0,0])} s\n")
        f.write(f"y distance of curve ends: {abs(curve_repara[-1,1] - curve_repara[0,1])} s\n")
        f.write(f"z distance of curve ends: {abs(curve_repara[-1,2] - curve_repara[0,2])} s\n")
        f.write(f"Areas of projections:\n")
        f.write(f"Area (xy-plane): {area_xy} s^2\n")
        f.write(f"Area (xz-plane): {area_xz} s^2\n")
        f.write(f"Area (yz-plane): {area_yz} s^2\n")

    # Save pulse sequence to a CSV file
    pulse_path = os.path.join(dir_path, 'pulse_sequence.bruker')
    np.savetxt(pulse_path, pulse_sequence, delimiter=',', fmt='%.12f', comments='')
    print("step 3/3 done")

    print(f"Saved curve to {curve_path}")
    print(f"Saved parameters and areas to {params_path}")
    print(f"Saved pulse sequence to {pulse_path}")
    return pulse_sequence

def test_pulse_sequence_from_curve():
    # Test parameters
    max_amplitude = 10000
    tau = 0.0000005
    dir_path = "/Users/leon/Desktop/Physik/Glaser/Bachelor_Thesis/other_data/pulse_sequence_from_curve"
    
    # Create a circular curve
    theta = np.linspace(0, 2 * np.pi, 100)
    radius = 1
    curve = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(curve[:,0], curve[:,1], curve[:,2], label='Curve')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('Curve')
    plt.show()

    # Run the function
    pulse_sequence = pulse_sequence_from_curve(curve, max_amplitude, tau, dir_path)
    space_curve = calculate_space_curve(pulse_sequence[:,0], pulse_sequence[:,1], 0, max_amplitude, tau)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(space_curve[:,0], space_curve[:,1], space_curve[:,2], label='Curve')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('Space-Curve from pulse sequence')
    plt.show()

#test_pulse_sequence_from_curve()
