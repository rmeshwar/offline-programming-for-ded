import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import traceback

# Define model paths 
base_model_path = "saved_models"
MODEL_PATHS = {
    "OLS Ridge OLD": {
        "model": os.path.join(base_model_path, 'OLS_ridge_OLD/neural_network_model.keras'),
        "scaler": os.path.join(base_model_path, 'OLS_ridge_OLD/scaler.pkl')
    },
    "OLS Ridge NODROPOUT OLD": {
        "model": os.path.join(base_model_path, 'OLS_ridge_NODROPOUT_OLD/neural_network_model.keras'),
        "scaler": os.path.join(base_model_path, 'OLS_ridge_NODROPOUT_OLD/scaler.pkl')
    },
    "DOE Ridge OLD": {
        "model": os.path.join(base_model_path, 'DOE_Ridge_OLD/neural_network_model.keras'),
        "scaler": os.path.join(base_model_path, 'DOE_Ridge_OLD/scaler.pkl')
    },
    "DOE Ridge NODROPOUT OLD": {
        "model": os.path.join(base_model_path, 'DOE_Ridge_NODROPOUT_OLD/neural_network_model.keras'),
        "scaler": os.path.join(base_model_path, 'DOE_Ridge_NODROPOUT_OLD/scaler.pkl')
    },
    "OLS Ridge": {
        "model": os.path.join(base_model_path, 'OLS_ridge/neural_network_model.keras'),
        "scaler": os.path.join(base_model_path, 'OLS_ridge/scaler.pkl')
    },
    "OLS Ridge NODROPOUT": {
        "model": os.path.join(base_model_path, 'OLS_ridge_NODROPOUT/neural_network_model.keras'),
        "scaler": os.path.join(base_model_path, 'OLS_ridge_NODROPOUT/scaler.pkl')
    },
    "DOE Ridge": {
        "model": os.path.join(base_model_path, 'DOE_Ridge/neural_network_model.keras'),
        "scaler": os.path.join(base_model_path, 'DOE_Ridge/scaler.pkl')
    },
    "DOE Ridge NODROPOUT": {
        "model": os.path.join(base_model_path, 'DOE_Ridge_NODROPOUT/neural_network_model.keras'),
        "scaler": os.path.join(base_model_path, 'DOE_Ridge_NODROPOUT/scaler.pkl')
    }
}

# Allowed value ranges
allowed_ranges = {
    'Travel Speed': (10, 60),
    'Wire Feed Speed': (3, 12),
    'Stepover Distance': (2, 12)
}

# Ideal values used for loss calculation
IDEAL_VALUES = {
    "Ratio of Valley Area to Bead Area": 0,
    "Ratio of Bead Heights": 1
}

# Define the specific test cases for (Travel Speed, Wire Feed Speed)
test_cases = [
    (10, 3),
    (15, 4),
    (50, 7.5),
    (55, 10),
    (60, 12)
]

# Generate stepover distances to test
stepover_distances = np.linspace(2, 12, 100)

# Open a log file to record high-loss cases
log_file_path = "logs/high_loss_log.txt"

with open(log_file_path, "w", encoding="utf-8") as log_file:
    log_file.write("High Loss Cases (Loss > 0.6)\n")
    log_file.write("=" * 60 + "\n")

    # Loop through each model
    for model_name, paths in MODEL_PATHS.items():
        model_path = paths["model"]
        scaler_path = paths["scaler"]

        # Ensure model and scaler exist
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"Skipping {model_name} - Missing files.")
            continue

        try:
            # Load the model and scaler
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)

            print(f"Testing {model_name}...")

            # Loop through the specific test cases
            for travel_speed, wire_feed_speed in test_cases:
                best_output_difference = float("inf")  # Track the lowest loss

                for stepover in stepover_distances:
                    input_data = np.array([[travel_speed, wire_feed_speed, stepover]])
                    input_data_df = pd.DataFrame(input_data, columns=['Travel Speed', 'Wire Feed Speed', 'Stepover Distance'])
                    input_data_scaled = scaler.transform(input_data_df)

                    predictions = model.predict(input_data_scaled)
                    pred_valley = predictions[0][0]
                    pred_bead_heights = predictions[0][1]

                    # Compute loss similar to GUI logic
                    loss = abs(pred_valley - IDEAL_VALUES["Ratio of Valley Area to Bead Area"]) + \
                           abs(pred_bead_heights - IDEAL_VALUES["Ratio of Bead Heights"])

                    # Track the lowest loss for this (travel_speed, wire_feed_speed) pair
                    best_output_difference = min(best_output_difference, loss)

                # If the lowest loss found is above 0.6, log it
                if best_output_difference > 0.6:
                    log_message = f"Model: {model_name}, Travel Speed: {travel_speed}, Wire Feed Speed: {wire_feed_speed}, Lowest Loss: {best_output_difference:.4f}\n"
                    print(log_message.strip())
                    log_file.write(log_message)

        except Exception as e:
            error_message = f"Error testing {model_name}: {str(e)}"
            print(error_message)
            log_file.write(f"{error_message}\n")
            traceback.print_exc()

print(f"\nTesting complete. High-loss cases logged in {log_file_path}")
