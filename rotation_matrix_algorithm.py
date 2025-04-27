import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from storing_saving_formatting import *


def rotation_matrix(axis, angle):   # returns the rotation matrix, that implements a rotation of angle around axis
    axis = axis / np.linalg.norm(axis)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    ux, uy, uz = axis
    # calculate rotation matrix
    R = np.array([
        [cos_a + ux**2 * (1 - cos_a), ux * uy * (1 - cos_a) - uz * sin_a, ux * uz * (1 - cos_a) + uy * sin_a],
        [uy * ux * (1 - cos_a) + uz * sin_a, cos_a + uy**2 * (1 - cos_a), uy * uz * (1 - cos_a) - ux * sin_a],
        [uz * ux * (1 - cos_a) - uy * sin_a, uz * uy * (1 - cos_a) + ux * sin_a, cos_a + uz**2 * (1 - cos_a)]
    ])
    return R

def calculate_space_curve(B1_amplitudes, phases, offset, max_Rabi_frequency, tau):
     # calculates the space curve using the rotation matrix method
    N = len(B1_amplitudes)
    R0 = [np.eye(3)]  # Initialize R0 with identity matrix for R0^{(0)}
    offset = 2 * np.pi * offset
    max_Rabi_frequency = 2 * np.pi * max_Rabi_frequency
    phases = phases *  np.pi / 180

    # Step 1: Calculate R0^{(k)} for k = 1,…,N
    for k in range(N):
        Rabi_frequency = max_Rabi_frequency * B1_amplitudes[k]/100
        Omega_dash = np.sqrt(Rabi_frequency**2 + offset**2)
        B_eff = np.array([Rabi_frequency * np.cos(phases[k]),
                         Rabi_frequency * np.sin(phases[k]),
                         offset])
        axis = B_eff/Omega_dash
        angle = Omega_dash * tau
        R0.append(rotation_matrix(axis, angle))
    
    # Step 2: Calculate R_k := R_0^{(k)} ... R_0^{(0)} for k = 1,...,N
    R_k = [R0[0]]  # Start with R0^{(0)} = Identity
    for k in range(1, N+1):
        R_k.append(np.dot(R0[k], R_k[k-1]))
    
    # Step 3: Calculate \dot{r}_k := R_k^T \cdot \hat{e}_z for k = 0,...,N
    e_z = np.array([0, 0, 1])
    r_dot = [np.dot(R.T, e_z) for R in R_k]
   
    # Step 4: Calculate r_k = sum(r_dot_{k-1} * tau) for k = 1,...,N+1
    r_values = [np.zeros(3)]  # Start with r_0 = 0
    for k in range(1, N+2):
        r_values.append(r_values[-1] + r_dot[k-1] * tau)
    
    return np.array(r_values)

def plot_space_curve(curve, new_figure = True, new_axes = True, show = True):
    if new_figure:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    elif (not new_figure and (not new_axes)):
        ax = plt.gca()
    else:
        fig = plt.gcf()
        # Move the first axes to the left
        plt.gca().set_position([0.0, 0.1, 0.4, 0.8])  # Move left by changing the 'left' position
        # Get the position of the current axes (ax1)
        current_axes = fig.get_axes()
        first_axes_position = current_axes[0].get_position()

        #    Calculate the new position for the second axes (to the right of the first one)
        new_left = first_axes_position.x1 + 0.05  # Add a gap of 0.05 between the axes
        new_bottom = first_axes_position.y0
        new_width = 0.4  # Same width as the first axes
        new_height = first_axes_position.height  # Same height as the first axes

        # Add the second axes to the right of the first axes
        ax = fig.add_axes([new_left, new_bottom, new_width, new_height], projection='3d')

    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2])#, marker='o')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set equal scaling for all axes
    if new_figure:
        max_range = np.array([curve[:, 0].max() - curve[:, 0].min(), 
                            curve[:, 1].max() - curve[:, 1].min(), 
                            curve[:, 2].max() - curve[:, 2].min()]).max() / 2.0

        mid_x = (curve[:, 0].max() + curve[:, 0].min()) * 0.5
        mid_y = (curve[:, 1].max() + curve[:, 1].min()) * 0.5
        mid_z = (curve[:, 2].max() + curve[:, 2].min()) * 0.5

        #ax.set_xlim(mid_x - max_range, mid_x + max_range)
        #ax.set_ylim(mid_y - max_range, mid_y + max_range)
        #ax.set_zlim(mid_z - max_range, mid_z + max_range)
    


    ax.ticklabel_format(style='plain')

    if show:
        plt.show()



"""
# Example usage(900◦x900◦y900◦x sequence):
pulse_sequence = np.array([[100,0],[100,0],[100,0],[100,0],[100,0],[100,0],[100,0],[100,0],[100,0],[100,0],[100,0],[100,0],[100,0],[100,0],[100,0],[100,0]])
offset = 0 # offset \nu_{off} in kHz
max_Rabi_frequency = 10000  # max_Rabi_frequency \nu_{max} in kHz
tau = 0.0000025  #sub-pulse duration in sec
space_curve = calculate_space_curve(pulse_sequence[:,0], pulse_sequence[:,1], offset, max_Rabi_frequency, tau)
plot_space_curve(space_curve)

# Example usage:
pulse_sequence = np.array([[100,180],
                           [100,180],
                           [100,180],
                           [100,-90],
                           [100,-90],
                           [100,-90],
                           [100,180],
                           [100,180],
                           [100,180]])
offset = 0 # offset \nu_{off} in kHz
max_Rabi_frequency = 1  # max_Rabi_frequency \nu_{max} in kHz
tau = 1/12  #sub-pulse duration in sec
space_curve = calculate_space_curve(pulse_sequence[:,0], pulse_sequence[:,1], offset, max_Rabi_frequency, tau)
plot_space_curve(space_curve)


pulse_sequence = getPulseSequence('/Users/leon/Desktop/Physik/Glaser/Analyse_und_Visualisierung_von_robusten_Kontrollpulsen/Pulssequenzen/UR_Pulse/UR36020kHz_30B1_rf10kHz/pulse1500.bruker')
offset = 12000 # offset \nu_{off} in kHz
max_Rabi_frequency = 10000  # max_Rabi_frequency \nu_{max} in kHz
tau = 0.0000005 #sub-pulse duration in sec
space_curve = calculate_space_curve(pulse_sequence[:,0], pulse_sequence[:,1], offset, max_Rabi_frequency, tau)
plot_space_curve(space_curve)
#"""