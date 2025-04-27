import os
import re
import numpy as np

def get_next_subdirectory(dir_path = "/Users/leon/Desktop/Physik/Glaser/Bachelor_Thesis/other_data/pulse_sequence_from_curve"):
    """
    Finds the next available numbered subdirectory in the given directory.

    Args:
        dir_path (str): Path to the parent directory.

    Returns:
        str: Path to the new subdirectory.
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"The directory {dir_path} does not exist.")

    # Gather existing subdirectories with numeric names
    existing_numbers = []
    for entry in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, entry)) and entry.isdigit():
            existing_numbers.append(int(entry))

    # Determine the next number
    next_number = max(existing_numbers, default=-1) + 1
    next_subdir = f"{next_number:03d}"  # Format as xxx (e.g., 003)

    # Full path to the new subdirectory
    next_subdir_path = os.path.join(dir_path, next_subdir)
    os.makedirs(next_subdir_path, exist_ok=True)

    return next_subdir_path

def getPulseSequence(PulseSequence):
    PS = Import_Bruker(PulseSequence)
    return PS

def Import_Bruker(filename):
    if isinstance(filename, str):
        with open(filename, "r") as f:
            text = f.read()
        text = re.sub(chr(13), chr(10), text)       
        text = re.sub(chr(10)+'{2}', chr(10), text)
        # Remove comments:
        #   char(10),
        #   then #,
        #   then any number of characters except char(10),
        #   then char(10)
        text = re.sub('\x0A#[^\x0A]*(?=\x0A)','', chr(10)+text+chr(10))
        PSlines = text.splitlines()
        while '' in PSlines:
            PSlines.remove('') 
        PS = np.empty((len(PSlines),2))
        for ind,line in enumerate(PSlines):
            PS[ind,:] = np.fromstring(PSlines[ind], dtype=float, sep=",")            #TODO: not gonna work??
        return np.array(PS)
    elif isinstance(filename, np.ndarray):
        PS = filename
        return PS
    else:
        raise Exception('unknown argument type for filename.')