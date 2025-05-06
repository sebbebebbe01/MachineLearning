"""
Simple script to convert the .mat file provided, for use in MATLAB, into a NumPy file for use in 
Python. Feel free to make any changes here as you see fit. This module does not need to be sent in
with the rest of the code.
"""

from scipy.io import loadmat
from numpy import save

# Replace 'your_file.mat' with the path to your .mat file
mat_file_path = 'C:/Users/sebas/.vscode/Maskininlärning/Assignment 1/A1_data.mat'

try:
    mat_data = loadmat(mat_file_path)
except FileNotFoundError:
    print(f"Error: File '{mat_file_path}' not found.")
    mat_data = None

if mat_data is not None:
    # Access variables from the .mat file
    # For example, if you have a variable named 'my_data' in the .mat file:
    t_test = mat_data['Ttest']
    t_train = mat_data['Ttrain']
    x = mat_data['X']
    x_audio = mat_data['Xaudio']
    x_interp = mat_data['Xinterp']
    fs = mat_data['fs']
    n = mat_data['n']
    n_interp = mat_data['ninterp']
    t = mat_data['t']
        
    # Replace 'output.npy' with the desired output file name
    npy_output_file = 'A1_data.npy'
        
    # Save the NumPy array as a .npy file
    with open(npy_output_file, "wb") as f:
        save(f, t_test)
        save(f, t_train)
        save(f, x)
        save(f, x_audio)
        save(f, x_interp)
        save(f, fs)
        save(f, n)
        save(f, n_interp)
        save(f, t)
    print(f"NumPy array saved as '{npy_output_file}'")