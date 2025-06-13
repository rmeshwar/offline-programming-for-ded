import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Load the data from the Excel file
file_path = 'data/New_DED.xlsx'
data = pd.read_excel(file_path)

# Create directories if they don't exist for LOOCV results
os.makedirs('logs/DOE_Ridge_LOOCV', exist_ok=True)

# Open a text file to save the LOOCV output
output_file = open('logs/DOE_Ridge_LOOCV/loocv_output.txt', 'w', encoding='utf-8')

# Redirect print statements to both console and text file
class DualOutput:
    def __init__(self, file1, file2):
        self.file1 = file1
        self.file2 = file2

    def write(self, data):
        self.file1.write(data)
        self.file2.write(data)

    def flush(self):
        pass

# Set sys.stdout to output to both console and file
sys.stdout = DualOutput(sys.stdout, output_file)

# Define preprocessing steps
poly = PolynomialFeatures(degree=2, include_bias=False)
scaler = MinMaxScaler(feature_range=(0, 1))

# Define independent variables (X) and dependent variables (y)
X = data[['Travel Speed', 'Wire Feed Speed', 'Stepover Distance']]
X_poly = poly.fit_transform(X)
X_scaled = scaler.fit_transform(X_poly)

dependent_variables = [
    'Ratio of Valley Area to Bead Area',
    'Ratio of Bead Heights',
    'Ratio of Upper Bead Height and Lowest Valley Point Height',
    'Ratio of Deposition Width to Lowest Valley Point Height'
]

# Perform LOOCV for each dependent variable
for y_var in dependent_variables:
    y = data[y_var]
    y_mean = np.mean(y)
    errors = []
    total_variance_y = sum((y - y_mean) ** 2)  # Total variance of the dependent variable

    print(f"\n{'=' * 80}")
    print(f"LOOCV for {y_var}")
    print(f"{'=' * 80}")

    n_samples = X_scaled.shape[0]

    for i in range(n_samples):  # Loop through each sample
        # Leave one sample out
        X_train = np.delete(X_scaled, i, axis=0)
        y_train = np.delete(y.values, i, axis=0)
        X_test = X_scaled[i].reshape(1, -1)
        y_test = y.values[i]

        # Train Ridge model
        ridge_model = Ridge(alpha=1.0).fit(X_train, y_train)

        # Predict the left-out sample
        y_pred = ridge_model.predict(X_test)

        # Calculate mean squared error for the left-out sample
        mse = mean_squared_error([y_test], y_pred)
        errors.append(mse)

    # Calculate LOOCV R² based on total variance explained
    r2_loocv = 1 - (sum(errors) / total_variance_y)

    # Output results
    print(f"Ridge LOOCV R² for {y_var}: {r2_loocv:.4f}")
    print(f"Ridge Mean LOOCV MSE for {y_var}: {np.mean(errors):.6f}\n")

# Close the output file
output_file.close()

# Reset sys.stdout to its original state
sys.stdout = sys.__stdout__
