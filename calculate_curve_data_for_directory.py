import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rotation_matrix_algorithm import getPulseSequence
from rotation_matrix_algorithm import calculate_space_curve

def calculate_closed_curve_area_app(CM2D, close_curve=False):
    if close_curve:
        CM2D = np.vstack((CM2D, CM2D[0]))  # closes curve if close_curve is True

    A = np.trapz(CM2D[:, 1], x=CM2D[:, 0])  # calculate the area using trapezoidal integration
    return abs(A)  # Return absolute area

# Main script
def process_bruker_files(dir_path, data_dir_path):
    offset = 0  # Offset in kHz
    max_Rabi_frequency = 10000  # Max Rabi frequency in kHz
    tau = 0.0000005  # Sub-pulse duration in seconds

    results = []
    file_names = []

    # Sort files alphabetically
    bruker_files = sorted(
        [f for f in os.listdir(dir_path) if f.startswith("pulse") and f.endswith(".bruker")]
    )

    for file_name in bruker_files:
        print(file_name)
        file_path = os.path.join(dir_path, file_name)
        pulse_sequence = getPulseSequence(file_path)

        # Calculate space curve
        space_curve = calculate_space_curve(pulse_sequence[:, 0], pulse_sequence[:, 1], offset, max_Rabi_frequency, tau)

        # Calculate distances
        x_dist = abs(space_curve[-1, 0] - space_curve[0, 0])
        y_dist = abs(space_curve[-1, 1] - space_curve[0, 1])
        z_dist = abs(space_curve[-1, 2] - space_curve[0, 2])

        # Calculate areas
        xy_area = calculate_closed_curve_area_app(space_curve[:, :2], close_curve=True)
        xz_area = calculate_closed_curve_area_app(space_curve[:, [0, 2]], close_curve=True)
        yz_area = calculate_closed_curve_area_app(space_curve[:, 1:], close_curve=True)

        # Append results
        results.append([x_dist, y_dist, z_dist, xy_area, xz_area, yz_area])
        file_names.append(file_name)

    # Create DataFrame
    columns = ["X_Distance", "Y_Distance", "Z_Distance", "XY_Area", "XZ_Area", "YZ_Area"]
    df = pd.DataFrame(results, columns=columns, index=file_names)

    # Save data to CSV
    data_csv_path = os.path.join(data_dir_path, "bruker_data.csv")
    df.to_csv(data_csv_path)

    # Plot data
    for column in columns:
        plt.figure()
        plt.scatter(range(len(df)), df[column], label=column)
        plt.title(f"Scatter Plot of {column}")
        plt.xlabel("File Index")
        plt.ylabel(column)
        plt.legend()
        plt.grid()

        # Save plot
        plot_path = os.path.join(data_dir_path, f"{column}_scatter.png")
        plt.savefig(plot_path)
        plt.close()

    print(f"Data and plots have been saved to {data_dir_path}")

#example:
"""
# Define paths
dir_path = "/Users/leon/Desktop/Physik/Glaser/Analyse_und_Visualisierung_von_robusten_Kontrollpulsen/Pulssequenzen/UR_Pulse/UR_ohne_B1_robustness_20_kHz/UR360_20kHz_noB1_rf10kHz(new_loop_sorted)"
data_dir_path = "/Users/leon/Desktop/Physik/Glaser/Bachelor_Thesis/other_data"
os.makedirs(data_dir_path, exist_ok=True)
data_dir_path = os.path.join(data_dir_path, "UR360_20kHz_noB1_rf10kHz(new_loop_sorted)")
os.mkdir(data_dir_path)

# Run the script
process_bruker_files(dir_path, data_dir_path)
"""