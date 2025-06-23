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
    #fid_data = compute_fidelity_plot(pulse_sequence, tau, max_amplitude, np.linspace(-400, 400, 40), np.linspace(0.95*max_amplitude, 1.05*max_amplitude, 40))

def lissajous_knot(max_amplitude = 10000, tau = 0.0000005, N = 1000, R = 1, a = 2, b = 3, c = 1, phi_a = np.pi/2, phi_b = np.pi/4, phi_c = 0, f = 1, dir_path = "/Users/leon/Desktop/Physik/Glaser/Bachelor_Thesis/other_data/pulse_sequence_from_curve"):
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
    
def clelia_curve(m = 7, n = 3.342, T = 8, N = 1000, max_amplitude = 10000, tau = 0.0000005, offset = 0, dir_path = "/Users/leon/Desktop/Physik/Glaser/Bachelor_Thesis/other_data/pulse_sequence_from_curve"):
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

def helix_straight_helix(N=1000,
                                max_amplitude=10000, tau=0.0000005, offset=0,
                                dir_path="/Users/leon/Desktop/Physik/Glaser/Bachelor_Thesis/other_data/pulse_sequence_from_curve"):
    """
    Generates a smooth 3D curve that has zero curvature at t=0.

    The curve is defined as:
        x(t) = t
        y(t) = t^3
        z(t) = sin(t)
    """
    R = 1
    h = 1
    omega = 1

    t_1 = np.linspace(0, 2*np.pi, round(N/3))
    t_2 = np.linspace(0, 10, round(N/3))
    #t_3 = np.linspace(3*np.pi, 5*np.pi, round(N/3))

    x_1 = R*np.cos(omega*t_1)
    y_1 = R*np.sin(omega*t_1)
    z_1 = h*t_1

    x_2 = 0*t_2 + x_1[-1]
    y_2 = -omega*R*t_2 + y_1[-1]
    z_2 = h*t_2 + z_1[-1]

    x_3 = x_1-x_1[0] + x_2[-1]
    y_3 = y_1-y_1[0] + y_2[-1]
    z_3 = z_1-z_2[0] + z_2[-1]

    x = np.hstack((x_1, x_2, x_3))
    y = np.hstack((y_1, y_2, y_3))
    z = np.hstack((z_1, z_2, z_3))

    curve = np.vstack((x, y, z)).T
    test_curve(curve, max_amplitude, tau, offset, dir_path)

def circle_straight_circle(N=1000,
                                max_amplitude=10000, tau=0.0000005, offset=0,
                                dir_path="/Users/leon/Desktop/Physik/Glaser/Bachelor_Thesis/other_data/pulse_sequence_from_curve"):
    """
    Generates a smooth 3D curve that has zero curvature at t=0.

    The curve is defined as:
        x(t) = t
        y(t) = t^3
        z(t) = sin(t)
    """
    R = 1
    omega = 1

    t_1 = np.linspace(0, 2*np.pi, round(N/3))
    t_2 = np.linspace(0, 10, round(N/3))
    #t_3 = np.linspace(3*np.pi, 5*np.pi, round(N/3))

    x_1 = R*np.cos(omega*t_1)
    y_1 = R*np.sin(omega*t_1)
    z_1 = h*t_1

    x_2 = 0*t_2 + x_1[-1]
    y_2 = -omega*R*t_2 + y_1[-1]
    z_2 = h*t_2 + z_1[-1]

    x_3 = x_1-x_1[0] + x_2[-1]
    y_3 = y_1-y_1[0] + y_2[-1]
    z_3 = z_1-z_2[0] + z_2[-1]

    x = np.hstack((x_1, x_2, x_3))
    y = np.hstack((y_1, y_2, y_3))
    z = np.hstack((z_1, z_2, z_3))

    curve = np.vstack((x, y, z)).T
    test_curve(curve, max_amplitude, tau, offset, dir_path)


def parity_curve_20(type = 1, N = 10000, max_amplitude = 10000, tau = 0.0000005, offset = 0, dir_path = "/Users/leon/Desktop/Physik/Glaser/Bachelor_Thesis/other_data/pulse_sequence_from_curve"):
    if (type == 1): #somehow doesnt work
        t = np.linspace(0,4*np.pi, N)
        x = np.sin(t/2)
        y = np.sin(t)*np.cos(t)**2
        z = np.sin(t)*np.cos(t)
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

def PS2curve2PS(PS, max_amplitude = 10000, tau = 0.0000005, offset = 0, dir_path = "/Users/leon/Desktop/Physik/Glaser/Bachelor_Thesis/other_data/pulse_sequence_from_curve"):
    PS = getPulseSequence(PS)
    curve = calculate_space_curve(PS[:,0], PS[:,1], offset, max_amplitude, tau)
    test_curve(curve, max_amplitude, tau, offset, dir_path)






#lissajous_knot()
#helix()
#clelia_curve()
#parity_curve_20()
#test_zero_curvature_curve()
#helix_straight_helix()
PS2curve2PS("/Users/leon/Desktop/Physik/Glaser/Analyse_und_Visualisierung_von_robusten_Kontrollpulsen/Pulssequenzen/UR_Pulse/UR36020kHz_30B1_rf10kHz/pulse1060.bruker")


#'/Users/leon/Desktop/Physik/Glaser/Analyse_und_Visualisierung_von_robusten_Kontrollpulsen/Pulssequenzen/testSequences/new_test/test_3.bruker'
#"/Users/leon/Desktop/Physik/Glaser/Analyse_und_Visualisierung_von_robusten_Kontrollpulsen/Pulssequenzen/UR_Pulse/UR36020kHz_30B1_rf10kHz/pulse1200.bruker"