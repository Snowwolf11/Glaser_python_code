import numpy as np
import matplotlib.pyplot as plt
import re
import numpy as np

from storing_saving_formatting import *
import time
    
def skew_symmetric_matrix(v):
    """
    Compute the skew-symmetric matrix for a vector v.
    """
    K = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    return K

def create_helix(starting_point, starting_tangent, starting_binormal, radius, pitch, arc_length, chirality, resolution):
   # Create a helix given the parameters.

    t = np.linspace(0, arc_length / np.sqrt((radius * 2 * np.pi)**2 + pitch**2), resolution)
    standard_helix = np.array([
        radius * np.cos(2 * np.pi * t * chirality),
        radius * np.sin(2 * np.pi * t * chirality),
        pitch * t
    ])
    
    # Compute the starting tangent of the standard helix
    starting_tangent_standard = np.array([0, radius * 2 * np.pi * chirality, pitch])
    starting_tangent_standard /= np.linalg.norm(starting_tangent_standard)
    
    # Compute the starting normal of the standard helix
    starting_normal_standard = np.array([-radius * 4 * np.pi**2, 0, 0])
    starting_normal_standard /= np.linalg.norm(starting_normal_standard)
    
    # Compute the starting binormal of the standard helix
    starting_binormal_standard = np.cross(starting_tangent_standard, starting_normal_standard)
    starting_binormal_standard /= np.linalg.norm(starting_binormal_standard)
    
    # Compute the ending tangent of the standard helix
    ending_tangent_standard = np.array([
        -radius * 2 * np.pi * chirality * np.sin(2 * np.pi * t[-1] * chirality),
        radius * 2 * np.pi * chirality * np.cos(2 * np.pi * t[-1] * chirality),
        pitch
    ])
    ending_tangent_standard /= np.linalg.norm(ending_tangent_standard)
    
    # Compute the ending normal of the standard helix
    ending_normal_standard = np.array([
        -radius * 4 * np.pi**2 * np.cos(2 * np.pi * t[-1] * chirality),
        -radius * 4 * np.pi**2 * np.sin(2 * np.pi * t[-1] * chirality),
        0
    ])
    ending_normal_standard /= np.linalg.norm(ending_normal_standard)
    
    # Compute the ending binormal of the standard helix
    ending_binormal_standard = np.cross(ending_tangent_standard, ending_normal_standard)
    ending_binormal_standard /= np.linalg.norm(ending_binormal_standard)
    
    starting_normal = np.cross(starting_binormal, starting_tangent)
    
    # Compute the rotation matrix to align the standard helix with the given vectors
    R1 = np.column_stack((starting_normal_standard, starting_binormal_standard, starting_tangent_standard)).T
    R2 = np.column_stack((starting_normal, starting_binormal, starting_tangent))
    rotation_matrix = np.dot(R2,R1)
    
    # Apply the rotation to the standard helix
    rotated_helix = (rotation_matrix @ standard_helix).T
    
    # Translate the helix to the starting point
    helix = rotated_helix + starting_point - rotated_helix[0]
    
    # Compute the ending tangent and binormal for the rotated helix
    ending_tangent = rotation_matrix @ ending_tangent_standard
    ending_tangent /= np.linalg.norm(ending_tangent)
    ending_binormal = rotation_matrix @ ending_binormal_standard
    ending_binormal /= np.linalg.norm(ending_binormal)
    
    return helix, ending_tangent, ending_binormal

def plot_space_curve(curve):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2])#, marker='o')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set equal scaling for all axes
    max_range = np.array([curve[:, 0].max() - curve[:, 0].min(), 
                          curve[:, 1].max() - curve[:, 1].min(), 
                          curve[:, 2].max() - curve[:, 2].min()]).max() / 2.0

    mid_x = (curve[:, 0].max() + curve[:, 0].min()) * 0.5
    mid_y = (curve[:, 1].max() + curve[:, 1].min()) * 0.5
    mid_z = (curve[:, 2].max() + curve[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.ticklabel_format(style='plain')

    plt.show()


def calculate_space_curve(B1_amplitudes, phases, offset, max_Rabi_frequency, tau, resolution = 1000):
     # calculates the space curve using the rotation matrix method
    N = len(B1_amplitudes)
    offset = 2 * np.pi * offset
    max_Rabi_frequency = 2 * np.pi * max_Rabi_frequency
    Rabi_frequencies = max_Rabi_frequency * B1_amplitudes/100
    phases = phases *  np.pi / 180

    starting_tangent = np.array([0, 0, 1])
    starting_binormal = np.array([-np.cos(phases[0]), -np.sin(phases[0]), 0])
    starting_point = np.array([0, 0, 0])

    helix_array = []

    # Create helices
    for k in range(N):
        Omega_k = Rabi_frequencies[k]
        phi_k = phases[k]
        B_eff = np.array([Omega_k * np.cos(phi_k), Omega_k * np.sin(phi_k), offset[k]])
        Omega_eff = np.linalg.norm(B_eff)
    
        radius = Omega_k / Omega_eff**2
        pitch = 2 * np.pi * offset[k] / Omega_eff**2
        arc_length = tau
    
        helix, ending_tangent, ending_binormal = create_helix(
            starting_point, starting_tangent, starting_binormal, radius, pitch, arc_length, -1, resolution
        )
    
        starting_point = helix[-1]
        starting_tangent = ending_tangent
        starting_binormal = ending_binormal
    
        if k < N - 1:
            theta = (phases[k + 1] - phases[k])
            K = skew_symmetric_matrix(starting_tangent)
            I = np.eye(3)
            R = I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
            starting_binormal = R @ starting_binormal
    
        helix_array.append(helix.T)

    # Combine to one curve
    return np.hstack(helix_array).T


################################################################
#Below: Optional version with non-constant offset and durations
################################################################

def _as_array(x, N, dtype=float):
    """
    Convert scalar input to length-N array.
    Leave array-like input as array.
    """
    arr = np.asarray(x, dtype=dtype)

    if arr.ndim == 0:
        arr = np.full(N, arr, dtype=dtype)

    if len(arr) != N:
        raise ValueError(f"Expected length {N}, got length {len(arr)}")

    return arr


def calculate_space_curve_variable_offset_tau(
    B1_amplitudes,
    phases,
    offset,
    max_Rabi_frequency,
    tau,
    resolution=1000,
):
    """
    Calculates the space curve using the helix method.

    Allows:
        - different B1 amplitudes per subpulse
        - different phases per subpulse
        - different offsets per subpulse
        - different durations tau per subpulse
        - different resolutions per subpulse

    Inputs:
        B1_amplitudes: array, percent of max_Rabi_frequency
        phases: array, degrees
        offset: scalar or array, Hz
        max_Rabi_frequency: Hz
        tau: scalar or array, seconds
        resolution: scalar or array, points per subpulse
    """

    B1_amplitudes = np.asarray(B1_amplitudes, dtype=float)
    phases = np.asarray(phases, dtype=float)

    N = len(B1_amplitudes)

    if len(phases) != N:
        raise ValueError("B1_amplitudes and phases must have same length")

    offset = _as_array(offset, N, dtype=float)
    tau = _as_array(tau, N, dtype=float)
    resolution = _as_array(resolution, N, dtype=int)

    offset = 2 * np.pi * offset
    max_Rabi_frequency = 2 * np.pi * max_Rabi_frequency
    Rabi_frequencies = max_Rabi_frequency * B1_amplitudes / 100
    phases = phases * np.pi / 180

    starting_tangent = np.array([0, 0, 1])
    starting_point = np.array([0, 0, 0])

    starting_binormal = np.array([
        -np.cos(phases[0]),
        -np.sin(phases[0]),
        0,
    ])

    helix_array = []

    for k in range(N):
        Omega_k = Rabi_frequencies[k]
        phi_k = phases[k]

        B_eff = np.array([
            Omega_k * np.cos(phi_k),
            Omega_k * np.sin(phi_k),
            offset[k],
        ])

        Omega_eff = np.linalg.norm(B_eff)

        radius = Omega_k / Omega_eff**2
        pitch = 2 * np.pi * offset[k] / Omega_eff**2
        arc_length = tau[k]

        helix, ending_tangent, ending_binormal = create_helix(
            starting_point=starting_point,
            starting_tangent=starting_tangent,
            starting_binormal=starting_binormal,
            radius=radius,
            pitch=pitch,
            arc_length=arc_length,
            chirality=-1,
            resolution=resolution[k],
        )

        starting_point = helix[-1]
        starting_tangent = ending_tangent
        starting_binormal = ending_binormal

        if k < N - 1:
            theta = phases[k + 1] - phases[k]

            K = skew_symmetric_matrix(starting_tangent)
            I = np.eye(3)

            R = I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

            starting_binormal = R @ starting_binormal
            starting_binormal /= np.linalg.norm(starting_binormal)

        helix_array.append(helix.T)

    return np.hstack(helix_array).T

"""
def test_compare_constant_and_variable_functions():


    pulse_sequence = np.array([
        [100,   180],
        [100,  270],
        [100,   180],
    ])

    B1_amplitudes = pulse_sequence[:, 0]
    phases = pulse_sequence[:, 1]

    offset = np.zeros(len(B1_amplitudes))       # Hz
    tau = np.full(len(B1_amplitudes), 25e-6)    # seconds

    max_Rabi_frequency = 10_000                 # Hz
    resolution = 2000

    curve_new = calculate_space_curve_variable_offset_tau(
        B1_amplitudes=B1_amplitudes,
        phases=phases,
        offset=offset,
        max_Rabi_frequency=max_Rabi_frequency,
        tau=tau,
        resolution=resolution,
    )*1e6

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        curve_new[:, 0],
        curve_new[:, 1],
        curve_new[:, 2],
        label="new variable offset/tau",
        linewidth=2,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    test_compare_constant_and_variable_functions()


def test_compare_constant_and_variable_functions():
    "" 
    Test case:
    Calculates the same pulse sequence with both helix functions
    and plots both curves in one 3D plot.
    ""

    pulse_sequence = np.array([
        [100,   0],
        [100,  90],
        [100, 180],
        [100, -90],
        [88.4,   13.2],
        [12,  113],
        [99.9, 181],
        [100, -90],
    ])

    B1_amplitudes = pulse_sequence[:, 0]
    phases = pulse_sequence[:, 1]

    offset = np.zeros(len(B1_amplitudes))       # Hz
    tau = np.full(len(B1_amplitudes), 25e-6)    # seconds

    max_Rabi_frequency = 10_000                 # Hz
    resolution = 200

    curve_old = calculate_space_curve(
        B1_amplitudes=B1_amplitudes,
        phases=phases,
        offset=offset,
        max_Rabi_frequency=max_Rabi_frequency,
        tau=tau[0],
        resolution=resolution,
    )

    curve_new = calculate_space_curve_variable_offset_tau(
        B1_amplitudes=B1_amplitudes,
        phases=phases,
        offset=offset,
        max_Rabi_frequency=max_Rabi_frequency,
        tau=tau,
        resolution=resolution,
    )

    print("curve_old shape:", curve_old.shape)
    print("curve_new shape:", curve_new.shape)

    endpoint_difference = np.linalg.norm(curve_old[-1] - curve_new[-1])
    print("endpoint difference:", endpoint_difference)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        curve_old[:, 0],
        curve_old[:, 1],
        curve_old[:, 2],
        label="old calculate_space_curve",
        linewidth=2,
    )

    ax.plot(
        curve_new[:, 0],
        curve_new[:, 1],
        curve_new[:, 2],
        "--",
        label="new variable offset/tau",
        linewidth=2,
    )

    ax.set_title("Old vs New Helix Space Curve")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    test_compare_constant_and_variable_functions()


# Example usage(900◦x900◦y900◦x sequence):
pulse_sequence = np.array([[100,0],[100,0]])
offset = 0 # offset \nu_{off} in kHz
max_Rabi_frequency = 10000  # max_Rabi_frequency \nu_{max} in kHz
tau = 0.000025  #sub-pulse duration in sec
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
offset = 0 # offset \nu_{off} in kHz
max_Rabi_frequency = 16700  # max_Rabi_frequency \nu_{max} in kHz
tau = 0.0000005 #sub-pulse duration in sec
tstart = time.time()
space_curve = calculate_space_curve(pulse_sequence[:,0], pulse_sequence[:,1], offset, max_Rabi_frequency, tau, resolution = 2)
tend = time.time()
#plot_space_curve(space_curve)
print(f"ellapsed time: {tend - tstart}")
"""