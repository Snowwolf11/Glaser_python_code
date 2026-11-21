import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from helix_algorithm import skew_symmetric_matrix
from storing_saving_formatting import *

def calculate_helix_segment_parameters_numpy(starting_point, starting_tangent, starting_binormal, radius, pitch, arc_length, chirality):
   # Create a helix given the parameters.
    resolution = 10000

    t_max = arc_length / np.sqrt((radius * 2 * np.pi)**2 + pitch**2)
    t = np.linspace(0, t_max, resolution)
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
    
    helix = rotated_helix + starting_point - rotated_helix[0]
    translation = starting_point - rotated_helix[0]

    # Compute the ending tangent and binormal for the rotated helix
    ending_tangent = rotation_matrix @ ending_tangent_standard
    ending_tangent /= np.linalg.norm(ending_tangent)
    ending_binormal = rotation_matrix @ ending_binormal_standard
    ending_binormal /= np.linalg.norm(ending_binormal)
    
    return [rotation_matrix, translation, t_max, radius, chirality, pitch], helix, ending_tangent, ending_binormal


def calculate_helix_segments_numpy(B1_amplitudes, phases, offset, max_Rabi_frequency, tau):
     # calculates the rotation_matrices, translations and t_maxfor all helix segments
    N = len(B1_amplitudes)
    offset = 2 * np.pi * offset
    max_Rabi_frequency = 2 * np.pi * max_Rabi_frequency
    Rabi_frequencies = max_Rabi_frequency * B1_amplitudes/100
    phases = phases *  np.pi / 180

    starting_tangent = np.array([0, 0, 1])
    starting_binormal = np.array([-np.cos(phases[1]), -np.sin(phases[1]), 0])
    starting_point = np.array([0, 0, 0])

    segment_parameters_array = [] #rotation_matrix, translation vector, t_max, radius, chirality, pitch

    # Create helices
    for k in range(N):
        Omega_k = Rabi_frequencies[k]
        phi_k = phases[k]
        B_eff = np.array([Omega_k * np.cos(phi_k), Omega_k * np.sin(phi_k), offset])
        Omega_eff = np.linalg.norm(B_eff)
    
        radius = Omega_k / Omega_eff**2
        pitch = 2 * np.pi * offset / Omega_eff**2
        arc_length = tau
    
        segment_parameters, helix, ending_tangent, ending_binormal = calculate_helix_segment_parameters_numpy(
            starting_point, starting_tangent, starting_binormal, radius, pitch, arc_length, -1
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
    
        segment_parameters_array.append(segment_parameters)
    return segment_parameters_array


def analytical_helix_segment_numpy(t, rotation_matrix, translation, radius, chirality, pitch):
    helix_point = (rotation_matrix @ np.array([
        radius * np.cos(2 * np.pi * t * chirality),
        radius * np.sin(2 * np.pi * t * chirality),
        pitch * t
    ])).T + translation
    return helix_point


def analytical_space_curve_numpy(t, segment_parameter_array):
    #segment_parameter_array is contains N arrays(N=numb of helix segments) that each contain [rotation_matrix, translation vector, t_max, radius, chirality, pitch]
    t_max_array = [segment_parameter[3] for segment_parameter in segment_parameter_array]
    t_array = np.insert(np.cumsum(t_max_array),0,0)
    segment_ind = np.searchsorted(t_array, t)-1
    t_segment = t-t_array[segment_ind]
    return analytical_helix_segment_numpy(t_segment, segment_parameter_array[segment_ind][0], segment_parameter_array[segment_ind][1], segment_parameter_array[segment_ind][3], segment_parameter_array[segment_ind][4], segment_parameter_array[segment_ind][5])

vectorized_analytical_space_curve = np.vectorize(
    analytical_space_curve_numpy,
    excluded=['segment_parameter_array'],  # pass segment_parameter_array only once
    signature='()->(n)'
)

def test_analytical_space_curve(pulse_sequence, offset, max_Rabi_frequency, tau):
    segment_parameter_array = calculate_helix_segments_numpy(pulse_sequence[:,0], pulse_sequence[:,1], offset, max_Rabi_frequency, tau)
    t_max_array = [segment_parameter[3] for segment_parameter in segment_parameter_array]
    t_array = np.insert(np.cumsum(t_max_array),0,0)
    t_end = t_array[-1]
    t_linspace = np.linspace(0,t_end, 1000)
    curve_points = vectorized_analytical_space_curve(t_linspace, segment_parameter_array=segment_parameter_array)

    # Plotting
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(curve_points[:,0], curve_points[:,1], curve_points[:,2], label='Analytical Helix Curve')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Analytical Space Curve made of Helix Segments')
    ax.legend()
    ax.grid(True)
    plt.show()

#######################################
#JNP version

def skew_symmetric_matrix_jnp(v):
    return jnp.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def calculate_helix_segment_parameters_jnp(starting_point, starting_tangent, starting_binormal, radius, pitch, arc_length, chirality):
    resolution = 10000

    t_max = arc_length / jnp.sqrt((radius * 2 * jnp.pi)**2 + pitch**2)
    t = jnp.linspace(0, t_max, resolution)

    standard_helix = jnp.array([
        radius * jnp.cos(2 * jnp.pi * t * chirality),
        radius * jnp.sin(2 * jnp.pi * t * chirality),
        pitch * t
    ])

    # Starting tangent of standard helix
    starting_tangent_standard = jnp.array([0, radius * 2 * jnp.pi * chirality, pitch])
    starting_tangent_standard /= jnp.linalg.norm(starting_tangent_standard)

    # Starting normal of standard helix
    starting_normal_standard = jnp.array([-radius * 4 * jnp.pi**2, 0, 0])
    starting_normal_standard /= jnp.linalg.norm(starting_normal_standard)

    # Starting binormal of standard helix
    starting_binormal_standard = jnp.cross(starting_tangent_standard, starting_normal_standard)
    starting_binormal_standard /= jnp.linalg.norm(starting_binormal_standard)

    # Ending tangent of standard helix
    t_last = t[-1]
    ending_tangent_standard = jnp.array([
        -radius * 2 * jnp.pi * chirality * jnp.sin(2 * jnp.pi * t_last * chirality),
        radius * 2 * jnp.pi * chirality * jnp.cos(2 * jnp.pi * t_last * chirality),
        pitch
    ])
    ending_tangent_standard /= jnp.linalg.norm(ending_tangent_standard)

    # Ending normal of standard helix
    ending_normal_standard = jnp.array([
        -radius * 4 * jnp.pi**2 * jnp.cos(2 * jnp.pi * t_last * chirality),
        -radius * 4 * jnp.pi**2 * jnp.sin(2 * jnp.pi * t_last * chirality),
        0
    ])
    ending_normal_standard /= jnp.linalg.norm(ending_normal_standard)

    # Ending binormal of standard helix
    ending_binormal_standard = jnp.cross(ending_tangent_standard, ending_normal_standard)
    ending_binormal_standard /= jnp.linalg.norm(ending_binormal_standard)

    starting_normal = jnp.cross(starting_binormal, starting_tangent)

    # Rotation matrix
    R1 = jnp.column_stack((starting_normal_standard, starting_binormal_standard, starting_tangent_standard)).T
    R2 = jnp.column_stack((starting_normal, starting_binormal, starting_tangent))
    rotation_matrix = R2 @ R1

    rotated_helix = (rotation_matrix @ standard_helix).T
    helix = rotated_helix + starting_point - rotated_helix[0]
    translation = starting_point - rotated_helix[0]

    # Ending tangent/binormal for rotated helix
    ending_tangent = rotation_matrix @ ending_tangent_standard
    ending_tangent /= jnp.linalg.norm(ending_tangent)
    ending_binormal = rotation_matrix @ ending_binormal_standard
    ending_binormal /= jnp.linalg.norm(ending_binormal)

    return [rotation_matrix, translation, t_max, radius, chirality, pitch], helix, ending_tangent, ending_binormal


def calculate_helix_segments_jnp(B1_amplitudes, phases, offset, max_Rabi_frequency, tau):
    offset = 2 * jnp.pi * offset
    max_Rabi_frequency = 2 * jnp.pi * max_Rabi_frequency
    Rabi_frequencies = max_Rabi_frequency * (jnp.asarray(B1_amplitudes) / 100.0)
    phases = jnp.asarray(phases) * jnp.pi / 180.0

    starting_tangent = jnp.array([0.0, 0.0, 1.0])
    starting_binormal = jnp.array([-jnp.cos(phases[1]), -jnp.sin(phases[1]), 0.0])
    starting_point = jnp.array([0.0, 0.0, 0.0])

    def body_fun(carry, k):
        starting_point, starting_tangent, starting_binormal, seg_params_list = carry

        Omega_k = Rabi_frequencies[k]
        phi_k = phases[k]
        B_eff = jnp.array([Omega_k * jnp.cos(phi_k), Omega_k * jnp.sin(phi_k), offset])
        Omega_eff = jnp.linalg.norm(B_eff)

        radius = Omega_k / Omega_eff**2
        pitch = 2 * jnp.pi * offset / Omega_eff**2
        arc_length = tau

        segment_parameters, helix, ending_tangent, ending_binormal = calculate_helix_segment_parameters_jnp(
            starting_point, starting_tangent, starting_binormal, radius, pitch, arc_length, -1
        )

        new_starting_point = helix[-1]
        new_starting_tangent = ending_tangent
        new_starting_binormal = ending_binormal

        def update_binormal(binormal):
            theta = (phases[k + 1] - phases[k])
            K = skew_symmetric_matrix_jnp(new_starting_tangent)
            I = jnp.eye(3)
            R = I + jnp.sin(theta) * K + (1 - jnp.cos(theta)) * (K @ K)
            return R @ binormal

        new_starting_binormal = jax.lax.cond(
            k < N - 1,
            lambda b: update_binormal(b),
            lambda b: b,
            new_starting_binormal
        )

        seg_params_list = seg_params_list + [segment_parameters]
        return (new_starting_point, new_starting_tangent, new_starting_binormal, seg_params_list), None

    (final_point, final_tangent, final_binormal, segment_parameters_array), _ = jax.lax.scan(
        body_fun,
        (starting_point, starting_tangent, starting_binormal, []),
        jnp.arange(N)
    )

    return segment_parameters_array


def analytical_helix_segment_jnp(t, rotation_matrix, translation, radius, chirality, pitch):
    helix_point = (rotation_matrix @ jnp.array([
        radius * jnp.cos(2 * jnp.pi * t * chirality),
        radius * jnp.sin(2 * jnp.pi * t * chirality),
        pitch * t
    ])).T + translation
    return helix_point


def analytical_space_curve_jnp(t, segment_parameter_array):
    t_max_array = jnp.array([segment[2] for segment in segment_parameter_array])
    t_array = jnp.insert(jnp.cumsum(t_max_array), 0, 0.0)

    segment_ind = jnp.searchsorted(t_array, t, side='right') - 1
    t_segment = t - t_array[segment_ind]

    # Use lax.switch to select the segment
    def get_segment(i):
        return segment_parameter_array[i]

    segment = jax.lax.switch(segment_ind, [lambda s=s: s for s in segment_parameter_array])

    return analytical_helix_segment_jnp(
        t_segment,
        segment[0],  # rotation_matrix
        segment[1],  # translation
        segment[3],  # radius
        segment[4],  # chirality
        segment[5]   # pitch
    )
"""
pulse_sequence = getPulseSequence('/Users/leon/Desktop/Physik/Glaser/Analyse_und_Visualisierung_von_robusten_Kontrollpulsen/Pulssequenzen/UR_Pulse/UR36020kHz_30B1_rf10kHz/pulse0100.bruker')
offset = 0 # offset \nu_{off} in kHz
max_Rabi_frequency = 10000  # max_Rabi_frequency \nu_{max} in kHz
tau = 0.0000005 #sub-pulse duration in sec
test_analytical_space_curve(pulse_sequence, offset, max_Rabi_frequency, tau)
"""
