import numpy as np

# Path to a single .npy file
file_path = 'data/processed_data/2_4_60bpm/2_4_60bpm_sample_1.npy'  # Replace with actual file path

# Load the data
data = np.load(file_path, allow_pickle=True)

# Check what kind of data is in the file
print(f"Data shape: {data.shape}")
print(f"First entry: {data[0]}")
