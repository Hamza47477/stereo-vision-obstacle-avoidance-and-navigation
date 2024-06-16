import pickle

# Load data from a .pkl file
with open(r"images\calibration_results\calibration_data.pkl", "rb") as f:
    calibration_data = pickle.load(f)

# Print the loaded data
for key, value in calibration_data.items():
    print(f"{key}: {value}\n")
