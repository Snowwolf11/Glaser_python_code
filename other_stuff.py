import numpy as np
import pandas as pd
import matplotlib as mplt
import matplotlib.pyplot as plt

#from calculate_curve_data_for_directory import *
#from curves_to_pulse_sequences import *
#from differential_geometry import *
from fidelity_plots import *
#from helix_algorithm import *
#from pulse_sequence_from_curve import *
#from rotation_matrix_algorithm import *
from storing_saving_formatting import *

max_amplitude = 10000
tau = 0.0000005
pulse_sequence = getPulseSequence("/Users/leon/Desktop/Physik/Glaser/Analyse_und_Visualisierung_von_robusten_Kontrollpulsen/Pulssequenzen/UR_Pulse/UR_ohne_B1_robustness_20_kHz/UR360_20kHz_noB1_rf10kHz(new_loop_sorted)/pulse0960.bruker")
fid_data = compute_fidelity_plot(pulse_sequence, tau, max_amplitude, np.linspace(-15000, 15000, 50), np.linspace(0.8*max_amplitude, 1.2*max_amplitude, 50))
