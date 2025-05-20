import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipeinc

from rotation_matrix_algorithm import *
from differential_geometry import *
from pulse_sequence_from_curve import *
from fidelity_plots import *

# Set print options to show all elements of large arrays
np.set_printoptions(threshold=np.inf)

def test_curve(curve, max_amplitude = 10000, tau = 0.0000005, offset = 0, dir_path = "/Users/leon/Desktop/Physik/Glaser/Bachelor_Thesis/other_data/pulse_sequence_from_curve"):
    pulse_sequence = pulse_sequence_from_curve(curve, max_amplitude, tau, dir_path)
    offset = 0
    space_curve = calculate_space_curve(pulse_sequence[:,0], pulse_sequence[:,1], offset, max_amplitude, tau)
    curve_1_scaled, curve_2_rotated, rmsd = compare_and_align_curves(curve, space_curve, plot=True)
    print(f"Similarity Score (RMSD) for different curves: {rmsd}")  
    fid_data = compute_fidelity_plot(pulse_sequence, tau, max_amplitude, np.linspace(-400, 400, 40), np.linspace(0.95*max_amplitude, 1.05*max_amplitude, 40))

def lissajous_knot(max_amplitude = 10000, tau = 0.0000005, N = 1000, R = 1, a = 1, b = 2, c = 3, phi_a = np.pi/2, phi_b = np.pi/4, phi_c = 0, f = 1, dir_path = "/Users/leon/Desktop/Physik/Glaser/Bachelor_Thesis/other_data/pulse_sequence_from_curve"):
    """
    Parametric equations of a scaled Lissajous knot.
    """
    #TODO: works worse under certain time tau. Prob to many point and numerical error. Maybee fix: calc for bigger tau and then change pulse sequence to fit smaller tau
    T_close = 2 * np.pi * np.lcm.reduce([a, b, c]) / (a * b * c * f)
    t = np.linspace(0,T_close,N)
    x = R * np.cos(a * f * t + phi_a)
    y = R * np.cos(b * f * t + phi_b)
    z = R * np.cos(c * f * t + phi_c)

    curve = np.vstack((x,y,z)).T
    test_curve(curve, max_amplitude, tau, dir_path)


def helix(r = 1, c = 1, T = 20, N = 1000, max_amplitude = 10000, tau = 0.0000005, offset = 0, dir_path = "/Users/leon/Desktop/Physik/Glaser/Bachelor_Thesis/other_data/pulse_sequence_from_curve"):
    """
    Computes the 3D coordinates for a helix.

    Parameters:
        r (float): Radius of the helix.
        c (float): Scaling factor for the z-axis.
        t (float or ndarray): Time parameter.

    Returns:
        tuple: (x, y, z) coordinates.
    """
    t = np.linspace(0,T,N)
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = c * t

    curve = np.vstack((x,y,z)).T
    test_curve(curve, max_amplitude, tau, offset, dir_path)
    
def clelia_curve(m = 2, n = 3, T = 20, N = 1000, max_amplitude = 10000, tau = 0.0000005, offset = 0, dir_path = "/Users/leon/Desktop/Physik/Glaser/Bachelor_Thesis/other_data/pulse_sequence_from_curve"):
    """
    Computes the 3D coordinates for a Clelia curve.

    Parameters:
        m (float): Frequency multiplier for theta.
        n (float): Frequency multiplier for phi.
        t (float or ndarray): Time parameter.

    Returns:
        tuple: (x, y, z) coordinates.
    """
    t = np.linspace(0,T,N)
    theta = m * t
    phi = n * t
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    curve = np.vstack((x,y,z)).T
    test_curve(curve, max_amplitude, tau, offset, dir_path)

def parity_curve_20(type = 1, N = 10000, max_amplitude = 10000, tau = 0.0000005, offset = 0, dir_path = "/Users/leon/Desktop/Physik/Glaser/Bachelor_Thesis/other_data/pulse_sequence_from_curve"):
    if (type == 1): #somehow doesnt work
        t = np.linspace(0,40*np.pi, N)
        x = 2000*np.sin(t/20)
        y = 2000*np.sin(t/10)*np.cos(t/10)**2
        z = 2000*np.sin(t/10)*np.cos(t/10)
        curve = np.vstack((x,y,z)).T
    elif (type == 2):
        t = np.linspace(0,2*np.pi, N)
        x = -1/2 * np.cos(t)**2
        y = 1/2 * (t - np.sin(t)*np.cos(t))
        z = np.sin(t)
        curve = np.vstack((x,y,z)).T
    elif (type == 3):
        t = np.linspace(0,2*np.pi, N)
        x = 1/4 * (np.sqrt(2)/2 * np.sin(2*t)- 2*np.sin(t))
        y = 1/4 * (np.sqrt(2)/2 * np.cos(2*t)+ 2*np.cos(t))
        z = np.sqrt(np.sqrt(2)+5/2)* ellipeinc(3*t/2, 2*np.sqrt(2)/(np.sqrt(2)+5/2))/3
        curve = np.vstack((x,y,z)).T
    elif (type == 4):
        t = np.linspace(0,2*np.pi, N)
        x = 1/6*(3*np.cos(t) - np.cos(3*t))
        y = 2/3 * np.sin(t)**3
        z = np.sin(t)
        curve = np.vstack((x,y,z)).T
    test_curve(curve, max_amplitude, tau, offset, dir_path)



#lissajous_knot()
#helix()
#clelia_curve()
parity_curve_20()