import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib
import traceback

# File paths
REAL_DATA_PATH = 'data/new_DED.xlsx'  # Experimental data
OLS_SYNTHETIC_DATA_PATH = 'data/synthetic/synthetic_data_ridge_500_samples.xlsx'
DOE_SYNTHETIC_DATA_PATH = 'data/synthetic/expanded_synthetic_data_500_samples.xlsx'

# Models paths
MODELS_PATHS = {
    "OLS_Ridge": {
        "model": 'saved_models/OLS_ridge/neural_network_model.keras',
        "scaler": 'saved_models/OLS_ridge/scaler.pkl'
    },
    "DOE_Ridge": {
        "model": 'saved_models/DOE_Ridge/neural_network_model.keras',
        "scaler": 'saved_models/DOE_Ridge/scaler.pkl'
    },
    "OLS_Ridge_NODROPOUT": {
        "model": 'saved_models/OLS_ridge_NODROPOUT/neural_network_model.keras',
        "scaler": 'saved_models/OLS_ridge_NODROPOUT/scaler.pkl'
    },
    "DOE_Ridge_NODROPOUT": {
        "model": 'saved_models/DOE_Ridge_NODROPOUT/neural_network_model.keras',
        "scaler": 'saved_models/DOE_Ridge_NODROPOUT/scaler.pkl'
    }
}

# Output log file
os.makedirs('logs/comparison', exist_ok=True)
OUTPUT_LOG = open('logs/comparison/model_comparison.txt', 'w', encoding='utf-8')

# Helper class for logging to both console and file
class DualOutput:
    def __init__(self, file1, file2):
        self.file1 = file1
        self.file2 = file2

    def write(self, data):
        self.file1.write(data)
        self.file2.write(data)

    def flush(self):
        pass

sys.stdout = DualOutput(sys.stdout, OUTPUT_LOG)

# Evaluation function
def evaluate_model(model_path, scaler_path, data_path, dataset_name):
    print(f"\nEvaluating model: {model_path}")
    print(f"Using dataset: {dataset_name}")
    print(f"Data file: {data_path}\n{'-'*50}")

    # Load data
    data = pd.read_excel(data_path)
    X = data[['Travel Speed', 'Wire Feed Speed', 'Stepover Distance']]
    y = data[['Ratio of Valley Area to Bead Area',
              'Ratio of Bead Heights',
              'Ratio of Upper Bead Height and Lowest Valley Point Height',
              'Ratio of Deposition Width to Lowest Valley Point Height']]

    # Load model and scaler
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    # Scale data
    X_scaled = scaler.transform(X)

    # Predict
    predictions = model.predict(X_scaled)

    # Calculate Mean Squared Error (MSE) for each target variable
    mse = np.mean((predictions - y.values)**2, axis=0)
    mae = np.mean(np.abs(predictions - y.values), axis=0)

    # Output comparison
    comparison_df = pd.DataFrame({
        'Actual Valley Area': y['Ratio of Valley Area to Bead Area'],
        'Predicted Valley Area': predictions[:, 0],
        'Actual Bead Heights': y['Ratio of Bead Heights'],
        'Predicted Bead Heights': predictions[:, 1],
        'Actual Upper Bead': y['Ratio of Upper Bead Height and Lowest Valley Point Height'],
        'Predicted Upper Bead': predictions[:, 2],
        'Actual Deposition Width': y['Ratio of Deposition Width to Lowest Valley Point Height'],
        'Predicted Deposition Width': predictions[:, 3],
    })

    print("Sample Predictions:")
    print(comparison_df.head(), "\n")
    print(f"MSE for each variable: {mse}")
    print(f"MAE for each variable: {mae}")
    return mse, mae

# Compare models on synthetic and real data
results = {}
datasets = {
    "OLS Synthetic Data": OLS_SYNTHETIC_DATA_PATH,
    "DOE Synthetic Data": DOE_SYNTHETIC_DATA_PATH,
    "Experimental Data": REAL_DATA_PATH
}

for model_name, paths in MODELS_PATHS.items():
    model_results = {}
    for dataset_name, dataset_path in datasets.items():
        mse, mae = evaluate_model(paths['model'], paths['scaler'], dataset_path, dataset_name)
        model_results[dataset_name] = {'MSE': mse, 'MAE': mae}
    results[model_name] = model_results

# Final results
print("\nComparison Results Summary:")
comparison_summary = pd.DataFrame(results)
print(comparison_summary.to_string())

# Additional analysis: Experimental Data performance
print("\nModel Performance on Experimental Data:")
experimental_results = {}
for model_name, paths in MODELS_PATHS.items():
    mse, mae = evaluate_model(paths['model'], paths['scaler'], REAL_DATA_PATH, "Experimental Data")
    experimental_results[model_name] = {'MSE': mse, 'MAE': mae}

experimental_comparison_df = pd.DataFrame(experimental_results).T
print(experimental_comparison_df.to_string(index=True))

# Save comparison results to a CSV file
comparison_summary.to_csv('logs/comparison/summary_results.csv')
experimental_comparison_df.to_csv('logs/comparison/experimental_results.csv')

OUTPUT_LOG.close()
