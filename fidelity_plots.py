import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from storing_saving_formatting import *

def compute_propagator(pulse_sequence, tau, nu_off, omega_max):
    """
    Compute the actual propagator U_F for a given pulse sequence, frequency offset,
    and maximum pulse amplitude. Each pulse is represented as a tuple (percentage, phi, duration).
    """
    U = np.eye(2, dtype=complex)  # Identity matrix as the starting propagator
    for percentage, phi in pulse_sequence:
        omega = percentage * omega_max/100  # Scale the Rabi frequency by percentage
        omega_prime = np.sqrt(omega**2 + (nu_off)**2)  # Generalized Rabi frequency
        
        # Components of the propagator matrix
        cos_term = np.cos(omega_prime * tau / 2)
        sin_term = np.sin(omega_prime * tau / 2)
        detuning_factor = (nu_off) / omega_prime
        rabi_factor = omega / omega_prime
        
        # Rotating frame propagator
        U_k = np.array([
            [cos_term - 1j * detuning_factor * sin_term, 
             1j * rabi_factor * sin_term * np.exp(-1j * phi*np.pi/180)],
            [1j * rabi_factor * sin_term * np.exp(1j * phi*np.pi/180), 
             cos_term + 1j * detuning_factor * sin_term]
        ])
        U = U_k @ U  # Sequentially apply the propagator
        #print("Unitarity error:", np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0])))
    
    return U

def compute_fidelity(U_T, U_F):
    """
    Compute the fidelity F = Re(<U_T|U_F>), where the inner product is the trace of U_T^\dagger U_F.
    """
    inner_product = np.trace(U_T.conj().T @ U_F)
    Q = np.real(inner_product / 2)  # Normalize by matrix size (2 for spin-1/2 system) // np.abs(inner_product)**2 / (d**2)  
    #print(Q)
    return np.log(1-Q)

def compute_fidelity_plot(pulse_sequence, sub_pulse_duration, maximum_amplitude, offset_range, amplitude_range, store_dir = None, U_T = None):
    # Parameters
    if (U_T is None):
        U_T = compute_propagator(pulse_sequence, sub_pulse_duration, 0, 2*np.pi*maximum_amplitude)  # Target propagator for a simple example (identity)
    print(f"Target evolution{U_T}")
    omega_max_range = 2*np.pi*amplitude_range
    omega_off_range = 2*np.pi*offset_range
    # Compute fidelity over the grid
    fidelity_data = []
    fidelity = np.zeros((len(omega_off_range), len(omega_max_range)))

    for i, nu_off in enumerate(omega_off_range):
        for j, omega_max in enumerate(omega_max_range):
            U_F = compute_propagator(pulse_sequence, sub_pulse_duration, nu_off, omega_max)
            fidelity[i, j] = compute_fidelity(U_T, U_F)
            fidelity_data.append({'nu_off': nu_off/(2*np.pi), 'max_amp': omega_max/(2*np.pi), 'fidelity': fidelity[i, j]})

    # Create a DataFrame
    df = pd.DataFrame(fidelity_data)

    # Save the DataFrame to a CSV file
    if (store_dir == None):
        output_file = os.path.join(get_next_subdirectory(),"fidelity_data.csv")
    else:
        output_file = os.path.join(get_next_subdirectory(store_dir),"fidelity_data.csv")
    df.to_csv(output_file, index=False)

    # Plotting the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(fidelity, extent=[omega_max_range[0]/(2000*np.pi), omega_max_range[-1]/(2000*np.pi), 
                omega_off_range[0]/(2000*np.pi), omega_off_range[-1]/(2000*np.pi)], origin='lower', 
                aspect='auto', cmap='jet')
    plt.colorbar(label='Fidelity')
    plt.xlabel('Max Pulse Amplitude [kHz]')
    plt.ylabel('Frequency Offset (ν_off) [kHz]')
    plt.title('Fidelity Heatmap')
    plt.show()

    return fidelity_data

def test_fidelity_plots():
    pulse_sequence = np.array([[100,0]])
    tau = 5*1e-7
    nu_off = 0
    omega_max = 10000
    U_calc = compute_propagator(pulse_sequence, tau, nu_off, omega_max)
    print(f"Calculated Evolution{U_calc}")
    expected_angle = omega_max * tau
    expected_U = np.array([
    [np.cos(expected_angle / 2), -1j * np.sin(expected_angle / 2)],
    [-1j * np.sin(expected_angle / 2), np.cos(expected_angle / 2)]
    ])
    print(f"Expected Evolution{expected_U}")

    pulse_sequence = getPulseSequence('/Users/leon/Desktop/Physik/Glaser/Analyse_und_Visualisierung_von_robusten_Kontrollpulsen/Pulssequenzen/UR_Pulse/UR36020kHz_30B1_rf10kHz/pulse1500.bruker')
    offset_range = np.linspace(-14000,14000,60) # offset \nu_{off} in kHz
    amplitude_range = np.linspace(6000,15000, 30)
    max_Amplitude = 10000  # in kHz
    tau = 0.0000005 #sub-pulse duration in sec
    store_dir = "/Users/leon/Desktop/Physik/Glaser/Exports"
    fid_data = compute_fidelity_plot(pulse_sequence, tau, max_Amplitude, offset_range, amplitude_range, store_dir = store_dir, U_T = - np.eye(2, dtype=complex))





#test_fidelity_plots()