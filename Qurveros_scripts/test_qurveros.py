import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import numpy as np
import jax.numpy as jnp

from qurveros.spacecurve import SpaceCurve
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from storing_saving_formatting import *
from rotation_matrix_algorithm import *
from calculate_analytical_curve import *


pulse_sequence = getPulseSequence('/Users/leon/Desktop/Physik/Glaser/Analyse_und_Visualisierung_von_robusten_Kontrollpulsen/Pulssequenzen/UR_Pulse/UR36020kHz_30B1_rf10kHz/pulse0250.bruker')
offset = 0 # offset \nu_{off} in kHz
max_Rabi_frequency = 10000  # max_Rabi_frequency \nu_{max} in kHz
tau = 0.0000005 #sub-pulse duration in sec
segment_parameter_array = calculate_helix_segments_numpy(pulse_sequence[:,0], pulse_sequence[:,1], offset, max_Rabi_frequency, tau)
# 2) Convert to JAX-friendly version (list of tuples with jnp arrays)
segment_parameter_array_jnp = [
    (
        jnp.asarray(seg[0], dtype=jnp.float64),  # rotation_matrix
        jnp.asarray(seg[1], dtype=jnp.float64),  # translation vector
        float(seg[2]),                           # t_max (scalar)
        float(seg[3]),                           # radius
        float(seg[4]),                           # chirality
        float(seg[5])                            # pitch
    )
    for seg in segment_parameter_array
]

# Build t_array in jnp for the interval
t_max_array = jnp.array([seg[2] for seg in segment_parameter_array_jnp])
t_array = jnp.insert(jnp.cumsum(t_max_array), 0, 0.0)
interval = [0.0, float(t_array[-1])]

# 4) Define curve function
def curve_fun(x, params):
    return analytical_space_curve_jnp(x, params)

# 5) Create SpaceCurve
spacecurve = SpaceCurve(
    curve=curve_fun,
    order=0,
    interval=interval,
    params=segment_parameter_array_jnp
)


# Evaluate Frenet dictionary
spacecurve.evaluate_frenet_dict()

# Access curvature and torsion arrays for inspection
#curvature = spacecurve.frenet_dict['curvature']
#torsion = spacecurve.frenet_dict['torsion']

#print("Curvature array:", curvature)
#print("Torsion array:", torsion)

spacecurve.evaluate_robustness_properties()

spacecurve.plot_position()
plt.gca().view_init(elev=90, roll=50)
plt.show()

spacecurve.evaluate_control_dict('XY')
spacecurve.plot_fields('full')
plt.gcf().set_dpi(100)
plt.show()

"""
num_points = 1000
t_values = jnp.linspace(interval[0], interval[1], num_points)

# Vectorized evaluation over t_values
curve_points = jax.vmap(curve_fun, in_axes=(0, None))(t_values, segment_parameter_array)

# Convert to numpy for plotting
curve_points_np = np.array(curve_points)  # shape: (1000, 3)

# Plot in 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(curve_points_np[:, 0], curve_points_np[:, 1], curve_points_np[:, 2], lw=2)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Space Curve")

plt.show()
"""