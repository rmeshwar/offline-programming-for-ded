
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

# Load the data from the Excel file
file_path = 'data/New_DED.xlsx'
data = pd.read_excel(file_path)

# Create directories if they don't exist for K-Fold CV results
os.makedirs('logs/DOE_Ridge_KFold', exist_ok=True)

# Open a text file to save the K-Fold CV output
output_file = open('logs/DOE_Ridge_KFold/kfold_output.txt', 'w', encoding='utf-8')

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

# Set up K-Fold Cross-Validation with k=5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform K-Fold CV for each dependent variable
for y_var in dependent_variables:
    y = data[y_var]
    y_mean = np.mean(y)
    errors = []
    total_variance_y = sum((y - y_mean) ** 2)  # Total variance of the dependent variable

    print(f"\n{'='*80}")
    print(f"K-Fold CV for {y_var}")
    print(f"{'='*80}")

    for train_index, test_index in kf.split(X_scaled):
        # Split the data into training and test sets
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train Ridge model
        ridge_model = Ridge(alpha=1.0).fit(X_train, y_train)

        # Predict the test set
        y_pred = ridge_model.predict(X_test)

        # Calculate mean squared error for the test set
        mse = mean_squared_error(y_test, y_pred)
        errors.append(mse)

    # Calculate K-Fold CV R² based on total variance explained
    r2_kfold = 1 - (sum(errors) / total_variance_y)

    # Output results
    print(f"Ridge K-Fold R² for {y_var}: {r2_kfold:.4f}")
    print(f"Ridge Mean K-Fold MSE for {y_var}: {np.mean(errors):.6f}\n")

# Close the output file
output_file.close()

# Reset sys.stdout to its original state
sys.stdout = sys.__stdout__
